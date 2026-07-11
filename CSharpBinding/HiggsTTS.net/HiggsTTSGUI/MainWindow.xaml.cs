using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;
using Higgs.net;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace HiggsTTSGUI
{
    public partial class MainWindow : Window
    {
        private HiggsTTS? _tts;
        private float[]? _pcm;
        private int[]? _cachedRefCodes;
        private string? _cachedRefWav;
        private string? _cachedRefText;
        private string? _modelPath;
        private bool _busy;
        private bool _inputTouched;

        // ── Playback ──
        private WaveOutEvent? _player;
        private PcmMemoryProvider? _pcmProvider;
        private DispatcherTimer? _playTimer;
        private bool _seeking;

        private const string PlaceholderText = "Enter text to synthesize here...";

        // ── Tag definitions (Higgs emotion/style/sfx/prosody tokens) ──────
        private static readonly (string label, string token)[] Tags =
        {
            ("(none)", ""),
            ("── Emotion ──", ""),
            ("  Elation",         "<|emotion:elation|>"),
            ("  Amusement",       "<|emotion:amusement|>"),
            ("  Enthusiasm",      "<|emotion:enthusiasm|>"),
            ("  Determination",   "<|emotion:determination|>"),
            ("  Pride",           "<|emotion:pride|>"),
            ("  Contentment",     "<|emotion:contentment|>"),
            ("  Affection",       "<|emotion:affection|>"),
            ("  Relief",          "<|emotion:relief|>"),
            ("  Contemplation",   "<|emotion:contemplation|>"),
            ("  Confusion",       "<|emotion:confusion|>"),
            ("  Surprise",        "<|emotion:surprise|>"),
            ("  Awe",             "<|emotion:awe|>"),
            ("  Longing",         "<|emotion:longing|>"),
            ("  Arousal",         "<|emotion:arousal|>"),
            ("  Anger",           "<|emotion:anger|>"),
            ("  Fear",            "<|emotion:fear|>"),
            ("  Disgust",         "<|emotion:disgust|>"),
            ("  Bitterness",      "<|emotion:bitterness|>"),
            ("  Sadness",         "<|emotion:sadness|>"),
            ("  Shame",           "<|emotion:shame|>"),
            ("  Helplessness",    "<|emotion:helplessness|>"),
            ("── Style ──", ""),
            ("  Singing",         "<|style:singing|>"),
            ("  Shouting",        "<|style:shouting|>"),
            ("  Whispering",      "<|style:whispering|>"),
            ("── SFX ──", ""),
            ("  Cough",           "<|sfx:cough|>"),
            ("  Laughter",        "<|sfx:laughter|>"),
            ("  Crying",          "<|sfx:crying|>"),
            ("  Screaming",       "<|sfx:screaming|>"),
            ("  Burping",         "<|sfx:burping|>"),
            ("  Humming",         "<|sfx:humming|>"),
            ("  Sigh",            "<|sfx:sigh|>"),
            ("  Sniff",           "<|sfx:sniff|>"),
            ("  Sneeze",          "<|sfx:sneeze|>"),
            ("── Prosody ──", ""),
            ("  Speed Very Slow", "<|prosody:speed_very_slow|>"),
            ("  Speed Slow",      "<|prosody:speed_slow|>"),
            ("  Speed Fast",      "<|prosody:speed_fast|>"),
            ("  Speed V. Fast",   "<|prosody:speed_very_fast|>"),
            ("  Pitch Low",       "<|prosody:pitch_low|>"),
            ("  Pitch High",      "<|prosody:pitch_high|>"),
            ("  Pause",           "<|prosody:pause|>"),
            ("  Long Pause",      "<|prosody:long_pause|>"),
            ("  Expressive High", "<|prosody:expressive_high|>"),
            ("  Expressive Low",  "<|prosody:expressive_low|>"),
        };

        // ── Dark title bar via DWM ──────────────────────────────────────
        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int val, int size);

        private void Window_SourceInitialized(object? sender, EventArgs e)
        {
            var hwnd = new WindowInteropHelper(this).Handle;
            int useDark = 1;
            // DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Win10 20H1+), fallback 19
            if (DwmSetWindowAttribute(hwnd, 20, ref useDark, 4) != 0)
                DwmSetWindowAttribute(hwnd, 19, ref useDark, 4);
        }

        public MainWindow()
        {
            InitializeComponent();
            PopulateTags();
            LoadConfig();
        }

        // ── Config persistence ────────────────────────────────────────────

        private string ConfigPath => System.IO.Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory, "higgs_gui_config.json");

        private void LoadConfig()
        {
            try
            {
                if (File.Exists(ConfigPath))
                {
                    var json = File.ReadAllText(ConfigPath);
                    var cfg = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                    if (cfg != null)
                    {
                        TxtModel.Text   = cfg.GetValueOrDefault("model", "");
                        TxtRefWav.Text  = cfg.GetValueOrDefault("ref_wav", "");
                        TxtTokenizer.Text = cfg.GetValueOrDefault("tokenizer", "");
                        TxtRefText.Text = cfg.GetValueOrDefault("ref_text", "");
                        TxtTemp.Text    = cfg.GetValueOrDefault("temp", "0.9");
                        TxtSeed.Text    = cfg.GetValueOrDefault("seed", "42");
                        var tag = cfg.GetValueOrDefault("tag", "");
                        for (int i = 0; i < CmbTag.Items.Count; i++)
                        {
                            if (CmbTag.Items[i] is ComboBoxItem item && item.Tag as string == tag)
                            { CmbTag.SelectedIndex = i; break; }
                        }
                    }
                }
            }
            catch { /* first run, use defaults */ }
            EnableSynthIfReady();
        }

        private void SaveConfig()
        {
            var tagItem = CmbTag.SelectedItem as ComboBoxItem;
            var cfg = new Dictionary<string, string>
            {
                ["model"]     = TxtModel.Text,
                ["ref_wav"]   = TxtRefWav.Text,
                ["tokenizer"] = TxtTokenizer.Text,
                ["ref_text"]  = TxtRefText.Text,
                ["temp"]     = TxtTemp.Text,
                ["seed"]     = TxtSeed.Text,
                ["tag"]      = tagItem?.Tag as string ?? "",
            };
            File.WriteAllText(ConfigPath,
                JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true }));
        }

        // ── Tag combo ─────────────────────────────────────────────────────

        private void PopulateTags()
        {
            foreach (var (label, token) in Tags)
            {
                var item = new ComboBoxItem { Content = label, Tag = token };
                if (token == "" && label.StartsWith("──"))
                    item.IsEnabled = false;
                CmbTag.Items.Add(item);
            }
            CmbTag.SelectedIndex = 0;
        }

        // ── Browse buttons ────────────────────────────────────────────────

        private void BrowseModel(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            { Filter = "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*" };
            if (dlg.ShowDialog() == true) TxtModel.Text = dlg.FileName;
        }

        private async void LoadModel_Click(object sender, RoutedEventArgs e)
        {
            SaveConfig();
            BtnLoad.IsEnabled = false;
            BtnLoad.Content = "...";
            bool ok = false;
            try
            {
                ok = await Task.Run(() => EnsureModel());
            }
            catch (Exception ex)
            {
                LblStatus.Text = $"Load error: {ex.Message}";
                LblStatus.Foreground = new SolidColorBrush(Color.FromRgb(0xff, 0x00, 0x00));
            }
            BtnLoad.Content = ok ? "Loaded" : "Load";
            BtnLoad.IsEnabled = !ok;
            if (ok)
            {
                var tokPath = TxtTokenizer.Text.Trim();
                if (tokPath.Length > 0)
                {
                    if (!_tts!.SetTokenizer(tokPath))
                        LblStatus.Text = "Tokenizer load failed, using built-in BPE.";
                }
                BtnUnload.IsEnabled = true;
                EnableSynthIfReady();
            }
        }

        private void UnloadModel_Click(object sender, RoutedEventArgs e)
        {
            _tts?.Dispose();
            _tts = null;
            _modelPath = null;
            _pcm = null;
            _cachedRefCodes = null;
            _cachedRefWav = null;
            _cachedRefText = null;
            BtnLoad.IsEnabled = true;
            BtnLoad.Content = "Load";
            BtnUnload.IsEnabled = false;
            BtnSynth.IsEnabled = false;
            BtnPlay.IsEnabled = false;
            BtnStop.IsEnabled = false;
            BtnSave.IsEnabled = false;
            LblStatus.Text = "Model unloaded.";
        }

        private void TxtInput_GotFocus(object sender, RoutedEventArgs e)
        {
            if (!_inputTouched && TxtInput.Text == PlaceholderText)
            {
                TxtInput.Text = "";
                _inputTouched = true;
            }
        }

        private void TxtInput_LostFocus(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(TxtInput.Text))
            {
                _inputTouched = false;
                TxtInput.Text = PlaceholderText;
            }
        }

        private void BrowseTokenizer(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            { Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*" };
            if (dlg.ShowDialog() == true) TxtTokenizer.Text = dlg.FileName;
        }

        private void BrowseRefWav(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            { Filter = "Audio files (*.wav;*.mp3;*.flac;*.ogg)|*.wav;*.mp3;*.flac;*.ogg|All files (*.*)|*.*" };
            if (dlg.ShowDialog() == true) TxtRefWav.Text = dlg.FileName;
        }

        private void EnableSynthIfReady()
        {
            BtnSynth.IsEnabled = !_busy && TxtModel.Text.Length > 0
                                  && TxtRefWav.Text.Length > 0;
        }

        // ── Load model ────────────────────────────────────────────────────

        private bool EnsureModel()
        {
            var path = "";
            Dispatcher.Invoke(() => path = TxtModel.Text.Trim());
            if (string.IsNullOrEmpty(path)) return false;

            if (_tts != null && path == _modelPath) return true;

            _tts?.Dispose();
            _tts = null;
            _modelPath = null;

            try
            {
                Dispatcher.Invoke(() =>
                {
                    LblStatus.Text = "Loading model...";
                    LblStatus.Foreground = new System.Windows.Media.SolidColorBrush(
                        System.Windows.Media.Color.FromRgb(0xff, 0xaa, 0x00));
                });
                _tts = new HiggsTTS(path);
                _modelPath = path;
                Dispatcher.Invoke(() =>
                {
                    LblStatus.Text = "Model loaded.";
                    LblStatus.Foreground = new System.Windows.Media.SolidColorBrush(
                        System.Windows.Media.Color.FromRgb(0x00, 0xaa, 0x00));
                });
                return true;
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    LblStatus.Text = $"Failed to load model: {ex.Message}";
                    LblStatus.Foreground = new System.Windows.Media.SolidColorBrush(
                        System.Windows.Media.Color.FromRgb(0xff, 0x00, 0x00));
                });
                return false;
            }
        }

        // ── Synthesize ────────────────────────────────────────────────────

        private async void Synthesize_Click(object sender, RoutedEventArgs e)
        {
            if (_busy) return;
            var text = TxtInput.Text.Trim();
            if (string.IsNullOrEmpty(text))
            {
                MessageBox.Show("Please enter text to synthesize.", "Notice");
                return;
            }

            SaveConfig();

            _busy = true;
            BtnSynth.IsEnabled = false;
            BtnPlay.IsEnabled = false;
            BtnStop.IsEnabled = false;
            BtnSave.IsEnabled = false;
            _pcm = null;
            StopPlayback();

            var refWav   = TxtRefWav.Text.Trim();
            var refText  = TxtRefText.Text.Trim();
            var temp     = float.TryParse(TxtTemp.Text, out var t) ? t : 0.9f;
            var seed     = int.TryParse(TxtSeed.Text, out var s) ? s : 42;
            var tagItem  = CmbTag.SelectedItem as ComboBoxItem;
            var tag      = tagItem?.Tag as string ?? "";
            var fullText = string.IsNullOrEmpty(tag) ? text : tag + text;

            BtnSynth.Content = "Synthesizing...";
            LblInfo.Content = "";

            // Auto-load model if not loaded
            if (_tts == null && TxtModel.Text.Trim().Length > 0)
            {
                LblStatus.Text = "Loading model...";
                bool ok = await Task.Run(() => EnsureModel());
                if (ok)
                {
                    var tokPath = TxtTokenizer.Text.Trim();
                    if (tokPath.Length > 0)
                        _tts!.SetTokenizer(tokPath);
                    BtnLoad.Content = "Loaded";
                    BtnLoad.IsEnabled = false;
                    BtnUnload.IsEnabled = true;
                }
                else
                {
                    _busy = false;
                    BtnSynth.Content = "Synthesize";
                    EnableSynthIfReady();
                    return;
                }
            }

            if (_tts == null)
            {
                _busy = false;
                BtnSynth.Content = "Synthesize";
                EnableSynthIfReady();
                return;
            }

            await Task.Run(() =>
            {
                try
                {
                    // ── Encode reference (cached per ref_wav + ref_text) ──
                    int[] refCodes;
                    if (_cachedRefCodes != null && _cachedRefWav == refWav && _cachedRefText == refText)
                    {
                        refCodes = _cachedRefCodes;
                    }
                    else
                    {
                        Dispatcher.Invoke(() =>
                            LblStatus.Text = "Encoding reference audio...");

                        var refAudio = ReadAudioMonoFloat(refWav);
                        refCodes = _tts!.EncodeRef(refAudio);
                        _cachedRefCodes = refCodes;
                        _cachedRefWav = refWav;
                        _cachedRefText = refText;
                    }

                    var sw = Stopwatch.StartNew();

                    Dispatcher.Invoke(() =>
                        LblStatus.Text = "Generating speech...");

                    var arRefText = string.IsNullOrEmpty(refText) ? null : refText;
                    var rawCodes = _tts.ARGenerate(fullText, arRefText, refCodes, temp, seed);

                    Dispatcher.Invoke(() =>
                        LblStatus.Text = "Decoding audio...");

                    var pcm = _tts.Decode(rawCodes);
                    _pcm = pcm;
                    sw.Stop();

                    var duration = pcm.Length / 24000.0;
                    var rtf = sw.Elapsed.TotalSeconds / duration;
                    Dispatcher.Invoke(() =>
                    {
                        LblStatus.Text = "Ready.";
                        LblInfo.Content = $"Duration {duration:F1}s | RTF {rtf:F2}x | {pcm.Length} samples";
                    });
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        LblInfo.Content = $"Error: {ex.Message}";
                        LblStatus.Text = "Error.";
                    });
                }
                finally
                {
                    Dispatcher.Invoke(() =>
                    {
                        _busy = false;
                        BtnSynth.Content = "Synthesize";
                        BtnPlay.IsEnabled = _pcm != null;
                        BtnStop.IsEnabled = _pcm != null;
                        BtnSave.IsEnabled = _pcm != null;
                        EnableSynthIfReady();
                    });
                }
            });
        }

        // ── Play / Pause / Stop ────────────────────────────────────────────

        private void PlayPause_Click(object sender, RoutedEventArgs e)
        {
            if (_pcm == null) return;
            try
            {
                if (_player != null && _player.PlaybackState == PlaybackState.Playing)
                {
                    // Pause
                    _player.Pause();
                    _playTimer?.Stop();
                    BtnPlay.Content = "▶";
                }
                else if (_player != null && _player.PlaybackState == PlaybackState.Paused)
                {
                    // Resume
                    _player.Play();
                    _playTimer?.Start();
                    BtnPlay.Content = "⏸";
                }
                else
                {
                    // Start new playback
                    StopPlayback();

                    _pcmProvider = new PcmMemoryProvider(_pcm, 24000);
                    _player = new WaveOutEvent();
                    _player.PlaybackStopped += (_, _) => Dispatcher.Invoke(OnPlaybackEnded);
                    _player.Init(_pcmProvider);
                    _player.Play();

                    _playTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
                    _playTimer.Tick += PlayTimer_Tick;
                    _playTimer.Start();

                    BtnPlay.Content = "⏸";
                    BtnStop.IsEnabled = true;
                    SldPosition.IsEnabled = true;
                }
            }
            catch (Exception ex)
            {
                LblInfo.Content = $"Playback error: {ex.Message}";
                StopPlayback();
            }
        }

        private void Stop_Click(object sender, RoutedEventArgs e)
        {
            StopPlayback();
        }

        private void StopPlayback()
        {
            _player?.Stop();
            _player?.Dispose();
            _player = null;
            _pcmProvider = null;
            _playTimer?.Stop();
            _playTimer = null;
            BtnPlay.Content = "▶";
            BtnStop.IsEnabled = _pcm != null;
            SldPosition.IsEnabled = false;
            SldPosition.Value = 0;
            LblTime.Text = FormatTime(0) + " / " + FormatTime(_pcm?.Length / 24000f ?? 0);
        }

        private void OnPlaybackEnded()
        {
            _playTimer?.Stop();
            BtnPlay.Content = "▶";
            BtnStop.IsEnabled = _pcm != null;
            SldPosition.Value = 0;
            SldPosition.IsEnabled = false;
            _player?.Dispose();
            _player = null;
            _pcmProvider = null;
        }

        private void PlayTimer_Tick(object? sender, EventArgs e)
        {
            if (_pcmProvider == null || _seeking) return;
            double sec = _pcmProvider.CurrentTime;
            double total = _pcmProvider.TotalTime;
            if (total > 0)
                SldPosition.Value = sec / total;
            LblTime.Text = FormatTime(sec) + " / " + FormatTime(total);
        }

        // ── Seek ───────────────────────────────────────────────────────────

        private void SldPosition_PreviewMouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _seeking = true;
            _player?.Pause();
        }

        private void SldPosition_PreviewMouseUp(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _seeking = false;
            if (_pcmProvider == null) return;
            double frac = SldPosition.Value;
            _pcmProvider.Seek((long)(frac * _pcmProvider.Length));
            if (_player != null && _player.PlaybackState == PlaybackState.Paused)
                _player.Play();
        }

        private void SldPosition_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!_seeking || _pcmProvider == null) return;
            double frac = SldPosition.Value;
            double sec = frac * _pcmProvider.TotalTime;
            LblTime.Text = FormatTime(sec) + " / " + FormatTime(_pcmProvider.TotalTime);
        }

        // ── Save ──────────────────────────────────────────────────────────

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            if (_pcm == null) return;
            var dlg = new Microsoft.Win32.SaveFileDialog
            {
                FileName = "output.wav",
                Filter = "WAV files (*.wav)|*.wav|All files (*.*)|*.*"
            };
            if (dlg.ShowDialog() != true) return;

            try
            {
                var wavBytes = PcmToWavBytes(_pcm, 24000);
                File.WriteAllBytes(dlg.FileName, wavBytes);
                LblInfo.Content = $"Saved: {System.IO.Path.GetFileName(dlg.FileName)}";
            }
            catch (Exception ex)
            {
                LblInfo.Content = $"Save error: {ex.Message}";
            }
        }

        // ── WAV helpers ───────────────────────────────────────────────────

        private static float[] ReadAudioMonoFloat(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();

            // OGG → pipe through ffmpeg to raw 16-bit PCM at 24kHz mono
            if (ext == ".ogg")
                return ReadViaFfmpeg(path);

            using var reader = new AudioFileReader(path);
            ISampleProvider source = reader;
            if (reader.WaveFormat.Channels > 1)
                source = new StereoToMonoSampleProvider(source) { LeftVolume = 0.5f, RightVolume = 0.5f };

            if (reader.WaveFormat.SampleRate != 24000)
            {
                var resampler = new MediaFoundationResampler(reader,
                    new WaveFormat(24000, 1));
                source = resampler.ToSampleProvider();
            }

            var list = new List<float>();
            var buf = new float[4096];
            int n;
            while ((n = source.Read(buf, 0, buf.Length)) > 0)
            {
                for (int i = 0; i < n; i++)
                    list.Add(buf[i]);
            }
            return list.ToArray();
        }

        private static float[] ReadViaFfmpeg(string path)
        {
            using var proc = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ffmpeg",
                    Arguments = $"-i \"{path}\" -f s16le -ac 1 -ar 24000 -loglevel quiet pipe:1",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                }
            };
            proc.Start();

            var raw = new List<byte>();
            var buf = new byte[8192];
            int n;
            while ((n = proc.StandardOutput.BaseStream.Read(buf, 0, buf.Length)) > 0)
                raw.AddRange(buf.Take(n));

            proc.WaitForExit();

            var samples = new float[raw.Count / 2];
            for (int i = 0; i < samples.Length; i++)
            {
                short v = (short)(raw[i * 2] | (raw[i * 2 + 1] << 8));
                samples[i] = v / 32768f;
            }
            return samples;
        }

        private static string FormatTime(double seconds)
        {
            int m = (int)(seconds / 60);
            int s = (int)(seconds % 60);
            return $"{m}:{s:D2}";
        }

        private static byte[] PcmToWavBytes(float[] pcm, int sampleRate)
        {
            int dataSize = pcm.Length * 2; // 16-bit
            using var ms = new MemoryStream(44 + dataSize);
            using var bw = new BinaryWriter(ms);

            // RIFF header
            bw.Write(new[] { 'R', 'I', 'F', 'F' });
            bw.Write(36 + dataSize);
            bw.Write(new[] { 'W', 'A', 'V', 'E' });

            // fmt chunk
            bw.Write(new[] { 'f', 'm', 't', ' ' });
            bw.Write(16);           // chunk size
            bw.Write((short)1);     // PCM
            bw.Write((short)1);     // mono
            bw.Write(sampleRate);
            bw.Write(sampleRate * 2); // byte rate
            bw.Write((short)2);     // block align
            bw.Write((short)16);    // bits per sample

            // data chunk
            bw.Write(new[] { 'd', 'a', 't', 'a' });
            bw.Write(dataSize);
            foreach (var s in pcm)
            {
                var v = Math.Clamp(s, -1f, 1f);
                bw.Write((short)(v * 32767));
            }

            return ms.ToArray();
        }

        // ── Cleanup ───────────────────────────────────────────────────────

        protected override void OnClosed(EventArgs e)
        {
            StopPlayback();
            SaveConfig();
            _tts?.Dispose();
            base.OnClosed(e);
        }
    }

    /// <summary>Streams float[] PCM as 16-bit WAV to NAudio WaveOut.</summary>
    internal sealed class PcmMemoryProvider : IWaveProvider
    {
        private readonly float[] _pcm;
        private int _pos;

        public PcmMemoryProvider(float[] pcm, int sampleRate)
        {
            _pcm = pcm;
            WaveFormat = new WaveFormat(sampleRate, 16, 1);
        }

        public WaveFormat WaveFormat { get; }

        public long Length => _pcm.Length;
        public float CurrentTime => _pos / (float)WaveFormat.SampleRate;
        public float TotalTime => _pcm.Length / (float)WaveFormat.SampleRate;

        public void Seek(long samplePos)
        {
            _pos = (int)Math.Clamp(samplePos, 0, _pcm.Length);
        }

        public int Read(byte[] buffer, int offset, int count)
        {
            int samples = Math.Min(count / 2, _pcm.Length - _pos);
            for (int i = 0; i < samples; i++)
            {
                float v = Math.Clamp(_pcm[_pos + i], -1f, 1f);
                short s = (short)(v * 32767);
                buffer[offset + i * 2] = (byte)(s & 0xFF);
                buffer[offset + i * 2 + 1] = (byte)((s >> 8) & 0xFF);
            }
            _pos += samples;
            return samples * 2;
        }
    }
}
