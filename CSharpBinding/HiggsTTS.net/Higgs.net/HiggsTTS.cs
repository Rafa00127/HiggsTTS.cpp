using System.Runtime.InteropServices;

namespace Higgs.net;

/// <summary>
/// Higgs TTS native wrapper. Loads higgs_tts.dll and exposes the three-stage
/// synthesis pipeline. All intermediate data (codes, PCM) is managed by the
/// caller — no hidden state in the native handle.
/// </summary>
public sealed class HiggsTTS : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    // ── P/Invoke ──────────────────────────────────────────────────────────

    private const string DllName = "higgs_tts.dll";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr higgs_tts_load(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void higgs_tts_free(IntPtr handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int higgs_tts_set_tokenizer(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int higgs_tts_encode_ref(
        IntPtr handle,
        float[] audio, int nSamples,
        int[] outCodes);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int higgs_tts_ar_generate(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string targetText,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? refText,
        int[] inCodes, int TIn,
        float temperature, int seed,
        int[] outCodes);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int higgs_tts_decode(
        IntPtr handle,
        int[] codes, int TRaw,
        float[] outPcm);

    // ── Constructor / Dispose ────────────────────────────────────────────

    public HiggsTTS(string modelPath)
    {
        _handle = higgs_tts_load(modelPath);
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to load model: {modelPath}");
    }

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            higgs_tts_free(_handle);
            _handle = IntPtr.Zero;
        }
        _disposed = true;
    }

    private void EnsureNotDisposed()
    {
        if (_disposed || _handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(HiggsTTS));
    }

    /// <summary>
    /// Optionally set an HF tokenizer.json path for exact token matching.
    /// Must be called before ARGenerate. Returns false on failure.
    /// </summary>
    public bool SetTokenizer(string tokenizerPath)
    {
        EnsureNotDisposed();
        return higgs_tts_set_tokenizer(_handle, tokenizerPath) == 0;
    }

    // ── Pipeline ─────────────────────────────────────────────────────────

    /// <summary>
    /// Stage 1: Encode reference audio → RVQ codes.
    /// Returns int[T_frames * 8] row-major (t0_q0..t0_q7, t1_q0..t1_q7).
    /// </summary>
    /// <param name="audio">24 kHz mono float32 samples</param>
    public int[] EncodeRef(float[] audio)
    {
        EnsureNotDisposed();
        int nSamples = audio.Length;
        int maxFrames = nSamples / 400 + 8;
        int[] buffer = new int[maxFrames * 8];

        int T = higgs_tts_encode_ref(_handle, audio, nSamples, buffer);
        if (T < 0)
            throw new InvalidOperationException("EncodeRef failed");

        Array.Resize(ref buffer, T * 8);
        return buffer;
    }

    /// <summary>
    /// Stage 2: Autoregressive generation from ref codes + text.
    /// Returns int[T_raw * 8] row-major.
    /// </summary>
    /// <param name="targetText">Text to synthesize</param>
    /// <param name="refText">Transcription matching the reference audio (may be null)</param>
    /// <param name="refCodes">Output from EncodeRef</param>
    public int[] ARGenerate(string targetText, string? refText, int[] refCodes,
                             float temperature = 0.9f, int seed = 42)
    {
        EnsureNotDisposed();
        int TIn = refCodes.Length / 8;
        int maxSteps = targetText.Length * 12 + 500;
        int[] buffer = new int[maxSteps * 8];

        int TRaw = higgs_tts_ar_generate(_handle, targetText, refText,
                                          refCodes, TIn, temperature, seed, buffer);
        if (TRaw < 0)
            throw new InvalidOperationException("ARGenerate failed");

        Array.Resize(ref buffer, TRaw * 8);
        return buffer;
    }

    /// <summary>
    /// Stage 3: Decode RVQ codes → 24 kHz float32 PCM.
    /// </summary>
    public float[] Decode(int[] codes)
    {
        EnsureNotDisposed();
        int TRaw = codes.Length / 8;
        int maxSamples = TRaw * 960 + 1024;
        float[] buffer = new float[maxSamples];

        int nSamples = higgs_tts_decode(_handle, codes, TRaw, buffer);
        if (nSamples < 0)
            throw new InvalidOperationException("Decode failed");

        Array.Resize(ref buffer, nSamples);
        return buffer;
    }
}
