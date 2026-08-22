using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Higgs.net;

/// <summary>
/// Sends synthesized audio to QQ as a voice message through a OneBot v11 HTTP
/// server (NapCat / LLOneBot / Lagrange). Audio bytes -&gt; WAV -&gt; base64 -&gt;
/// record segment ("base64://&lt;wav&gt;") -&gt; POST send_private_msg / send_group_msg.
/// The OneBot server converts the WAV to Tencent SILK internally.
/// </summary>
public static class QqVoice
{
    /// <summary>OneBot v11 HTTP server default address (NapCat default port).</summary>
    public const string DefaultBaseUrl = "http://127.0.0.1:3000";

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(60) };

    /// <summary>
    /// Converts 24 kHz mono float32 PCM (as returned by <see cref="HiggsTTS.Decode"/>)
    /// into a 16-bit PCM WAV byte array.
    /// </summary>
    public static byte[] Float32PcmToWav(float[] pcm, int sampleRate = 24000)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));

        var data = new byte[pcm.Length * 2];
        for (int i = 0; i < pcm.Length; i++)
        {
            float s = pcm[i];
            short v = 0;
            if (float.IsFinite(s))
            {
                s = Math.Clamp(s, -1f, 1f);
                v = (short)MathF.Round(s * 32767f);
            }
            data[i * 2] = (byte)(v & 0xFF);
            data[i * 2 + 1] = (byte)((v >> 8) & 0xFF);
        }

        var ms = new MemoryStream(44 + data.Length);
        using var w = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        void Ascii(string s) => w.Write(Encoding.ASCII.GetBytes(s));

        Ascii("RIFF");
        w.Write((uint)(36 + data.Length));
        Ascii("WAVE");
        Ascii("fmt ");
        w.Write((uint)16);            // fmt chunk size
        w.Write((ushort)1);           // linear PCM
        w.Write((ushort)1);           // mono
        w.Write((uint)sampleRate);
        w.Write((uint)(sampleRate * 2)); // byte rate (mono, 16-bit)
        w.Write((ushort)2);           // block align
        w.Write((ushort)16);          // bits per sample
        Ascii("data");
        w.Write((uint)data.Length);
        w.Write(data);

        w.Flush();
        ms.Position = 0;
        return ms.ToArray();
    }

    /// <summary>
    /// Posts a OneBot v11 action and returns the raw JSON response body.
    /// Throws on non-2xx HTTP or a OneBot <c>status: failed</c> response.
    /// </summary>
    /// <param name="action">Action name, e.g. <c>send_private_msg</c>.</param>
    /// <param name="payload">Action body (user_id/group_id, message, ...).</param>
    /// <param name="baseUrl">OneBot HTTP base URL; defaults to <see cref="DefaultBaseUrl"/>.</param>
    /// <param name="token">Optional access token (NapCat http-server token).</param>
    public static async Task<string> SendActionAsync(
        string action,
        object payload,
        string? baseUrl = null,
        string? token = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(action);
        ArgumentNullException.ThrowIfNull(payload);

        var url = $"{baseUrl?.TrimEnd('/') ?? DefaultBaseUrl}/{action}";
        var json = JsonSerializer.Serialize(payload);
        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

        using var response = await Http.SendAsync(request, ct);
        var body = await response.Content.ReadAsStringAsync(ct);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"OneBot {action} failed: {(int)response.StatusCode} {body}");

        try
        {
            using var doc = JsonDocument.Parse(body);
            if (doc.RootElement.TryGetProperty("status", out var status)
                && status.ValueKind == JsonValueKind.String
                && status.GetString() == "failed")
                throw new HttpRequestException($"OneBot {action} failed: {body}");
        }
        catch (JsonException)
        {
            // non-JSON / empty body; nothing to check
        }

        return body;
    }

    /// <summary>OneBot v11 record segment whose file is base64 WAV.</summary>
    private static object RecordSegment(byte[] wav)
        => new
        {
            type = "record",
            data = new { file = "base64://" + Convert.ToBase64String(wav) },
        };

    /// <summary>Sends WAV bytes to a friend as a QQ voice message.</summary>
    public static async Task<string> SendPrivateVoiceAsync(
        long userId,
        byte[] wav,
        string? baseUrl = null,
        string? token = null,
        CancellationToken ct = default)
        => await SendActionAsync(
            "send_private_msg",
            new { user_id = userId, message = new[] { RecordSegment(wav) } },
            baseUrl, token, ct);

    /// <summary>Sends WAV bytes to a group as a QQ voice message.</summary>
    public static async Task<string> SendGroupVoiceAsync(
        long groupId,
        byte[] wav,
        string? baseUrl = null,
        string? token = null,
        CancellationToken ct = default)
        => await SendActionAsync(
            "send_group_msg",
            new { group_id = groupId, message = new[] { RecordSegment(wav) } },
            baseUrl, token, ct);

    /// <summary>Sends 24 kHz mono float32 PCM to a friend as a QQ voice message.</summary>
    public static async Task<string> SendPrivatePcmVoiceAsync(
        long userId,
        float[] pcm24kMono,
        string? baseUrl = null,
        string? token = null,
        CancellationToken ct = default)
        => await SendPrivateVoiceAsync(userId, Float32PcmToWav(pcm24kMono), baseUrl, token, ct);

    /// <summary>Sends 24 kHz mono float32 PCM to a group as a QQ voice message.</summary>
    public static async Task<string> SendGroupPcmVoiceAsync(
        long groupId,
        float[] pcm24kMono,
        string? baseUrl = null,
        string? token = null,
        CancellationToken ct = default)
        => await SendGroupVoiceAsync(groupId, Float32PcmToWav(pcm24kMono), baseUrl, token, ct);
}

/// <summary>
/// One-call voice cloning: a line of dialogue is synthesized in the reference
/// speaker's voice and sent to QQ as a voice message.
/// </summary>
public sealed class QqCharacterVoice
{
    private readonly HiggsTTS _tts;
    private readonly string? _baseUrl;
    private readonly string? _token;

    public QqCharacterVoice(HiggsTTS tts, string? baseUrl = null, string? token = null)
    {
        _tts = tts ?? throw new ArgumentNullException(nameof(tts));
        _baseUrl = baseUrl;
        _token = token;
    }

    private async Task<float[]> SynthesizeAsync(
        string dialogue,
        float[] refAudio,
        string? refText,
        float temperature,
        int seed,
        CancellationToken ct)
    {
        int[] refCodes = await Task.Run(() => _tts.EncodeRef(refAudio), ct);
        int[] codes = await Task.Run(
            () => _tts.ARGenerate(dialogue, refText, refCodes, temperature, seed), ct);
        return await Task.Run(() => _tts.Decode(codes), ct);
    }

    /// <summary>Synthesizes the dialogue and sends it to a friend as a QQ voice message.</summary>
    public async Task<string> SpeakPrivateAsync(
        long qq,
        string dialogue,
        float[] refAudio,
        string? refText = null,
        float temperature = 0.9f,
        int seed = 42,
        CancellationToken ct = default)
    {
        float[] pcm = await SynthesizeAsync(dialogue, refAudio, refText, temperature, seed, ct);
        return await QqVoice.SendPrivatePcmVoiceAsync(qq, pcm, _baseUrl, _token, ct);
    }

    /// <summary>Synthesizes the dialogue and sends it to a group as a QQ voice message.</summary>
    public async Task<string> SpeakGroupAsync(
        long groupId,
        string dialogue,
        float[] refAudio,
        string? refText = null,
        float temperature = 0.9f,
        int seed = 42,
        CancellationToken ct = default)
    {
        float[] pcm = await SynthesizeAsync(dialogue, refAudio, refText, temperature, seed, ct);
        return await QqVoice.SendGroupPcmVoiceAsync(groupId, pcm, _baseUrl, _token, ct);
    }
}