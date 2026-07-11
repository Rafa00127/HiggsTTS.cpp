using System.IO;
using System.Media;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Higgs.net;

namespace HiggsTTSGUI
{
    public partial class MainWindow : Window
    {
        private HiggsTTS? _tts;
        private float[]? _pcm;
        private string? _modelPath;
        private bool _busy;

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
            BtnLoad.IsEnabled = true;
            BtnLoad.Content = "Load";
            BtnUnload.IsEnabled = false;
            BtnSynth.IsEnabled = false;
            BtnPlay.IsEnabled = false;
            BtnSave.IsEnabled = false;
            LblStatus.Text = "Model unloaded.";
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
            { Filter = "WAV files (*.wav)|*.wav|All files (*.*)|*.*" };
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
            BtnSave.IsEnabled = false;
            _pcm = null;

            var refWav   = TxtRefWav.Text.Trim();
            var refText  = TxtRefText.Text.Trim();
            var temp     = float.TryParse(TxtTemp.Text, out var t) ? t : 0.9f;
            var seed     = int.TryParse(TxtSeed.Text, out var s) ? s : 42;
            var tagItem  = CmbTag.SelectedItem as ComboBoxItem;
            var tag      = tagItem?.Tag as string ?? "";
            var fullText = string.IsNullOrEmpty(tag) ? text : tag + text;

            BtnSynth.Content = "Synthesizing...";
            LblInfo.Content = "";

            // Load model on UI thread (needed before background work)
            LblStatus.Text = "Loading model...";
            bool modelOk = await Task.Run(() => EnsureModel());
            if (!modelOk)
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
                    Dispatcher.Invoke(() =>
                        LblStatus.Text = "Encoding reference audio...");

                    var refAudio = ReadWavMonoFloat(refWav);
                    var refCodes = _tts!.EncodeRef(refAudio);

                    Dispatcher.Invoke(() =>
                        LblStatus.Text = "Generating speech...");

                    var arRefText = string.IsNullOrEmpty(refText) ? null : refText;
                    var rawCodes = _tts.ARGenerate(fullText, arRefText, refCodes, temp, seed);

                    Dispatcher.Invoke(() =>
                        LblStatus.Text = "Decoding audio...");

                    var pcm = _tts.Decode(rawCodes);
                    _pcm = pcm;

                    var duration = pcm.Length / 24000.0;
                    Dispatcher.Invoke(() =>
                    {
                        LblStatus.Text = "Ready.";
                        LblInfo.Content = $"Duration {duration:F1}s | {pcm.Length} samples";
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
                        BtnSave.IsEnabled = _pcm != null;
                        EnableSynthIfReady();
                    });
                }
            });
        }

        // ── Play ──────────────────────────────────────────────────────────

        private void Play_Click(object sender, RoutedEventArgs e)
        {
            if (_pcm == null) return;
            try
            {
                var wavBytes = PcmToWavBytes(_pcm, 24000);
                using var ms = new MemoryStream(wavBytes);
                using var player = new SoundPlayer(ms);
                player.Play();
            }
            catch (Exception ex)
            {
                LblInfo.Content = $"Playback error: {ex.Message}";
            }
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

        private static float[] ReadWavMonoFloat(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            // RIFF header
            br.ReadChars(4); // "RIFF"
            br.ReadInt32();  // file size - 8
            br.ReadChars(4); // "WAVE"

            short bitsPerSample = 16;
            short numChannels = 1;
            int sampleRate = 24000;
            int dataSize = 0;

            // Find fmt and data chunks
            while (fs.Position < fs.Length)
            {
                var chunkId = new string(br.ReadChars(4));
                var chunkSize = br.ReadInt32();

                if (chunkId == "fmt ")
                {
                    br.ReadInt16(); // audio format (1=PCM)
                    numChannels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byte rate
                    br.ReadInt16(); // block align
                    bitsPerSample = br.ReadInt16();
                    if (chunkSize > 16) br.ReadBytes(chunkSize - 16);
                }
                else if (chunkId == "data")
                {
                    dataSize = chunkSize;
                    break;
                }
                else
                {
                    br.ReadBytes(chunkSize);
                }
            }

            int bytesPerSample = bitsPerSample / 8;
            int totalSamples = dataSize / bytesPerSample;
            var samples = new float[totalSamples / numChannels];

            for (int i = 0; i < samples.Length; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < numChannels; ch++)
                {
                    float v = bitsPerSample == 16
                        ? br.ReadInt16() / 32768f
                        : br.ReadByte() / 128f - 1f;
                    sum += v;
                }
                samples[i] = sum / numChannels;
            }

            // Resample to 24kHz if needed
            if (sampleRate != 24000)
                samples = ResampleLinear(samples, sampleRate, 24000);

            return samples;
        }

        private static float[] ResampleLinear(float[] src, int srcRate, int dstRate)
        {
            double ratio = (double)dstRate / srcRate;
            var dst = new float[(int)(src.Length * ratio)];
            for (int i = 0; i < dst.Length; i++)
            {
                double srcIdx = i / ratio;
                int idx0 = (int)srcIdx;
                int idx1 = Math.Min(idx0 + 1, src.Length - 1);
                double frac = srcIdx - idx0;
                dst[i] = (float)(src[idx0] * (1 - frac) + src[idx1] * frac);
            }
            return dst;
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
            SaveConfig();
            _tts?.Dispose();
            base.OnClosed(e);
        }
    }
}
