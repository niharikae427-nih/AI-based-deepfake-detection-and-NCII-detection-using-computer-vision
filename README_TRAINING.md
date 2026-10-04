# SENTRA AI — Real Model Training Package

यह package आपके SENTRA AI app के dummy detection को असली trained model से replace
करने के लिए है।

## इसमें क्या है

| File | Purpose |
|---|---|
| `sentra-ai-deepfake-training.ipynb` | Kaggle पर चलाने वाली training notebook — FaceForensics++ + DFDC दोनों datasets से face crops निकालकर एक असली Xception classifier train करती है |
| `image_detector_real.py` | `app/models/image_detector.py` की जगह लगाने वाली file — trained weights मिलते ही असली predictions देगी, ना मिलने पर automatically dummy fallback पर चली जाएगी |

## Step-by-step

### 1. Notebook Kaggle पर चलाएं
1. https://kaggle.com पर नया Notebook बनाएं और यह `.ipynb` file upload करें।
2. **Add Data** से जोड़ें:
   - Kaggle competition dataset: **Deepfake Detection Challenge (DFDC)**
   - कोई भी community-uploaded **FaceForensics++** dataset (Kaggle Datasets में search करें)
3. Settings → Accelerator → **GPU** on करें (free quota मिलता है)।
4. Notebook के "CONFIG" cell में paths को अपने attached datasets के नामों से match करें।
5. **Run All** दबाएं। पहला run test के लिए छोटा रखा गया है (कम videos, 8 epochs) — यह
   करीब 1-2 घंटे लग सकते हैं। ठीक चलने पर `MAX_VIDEOS_PER_CLASS` और `EPOCHS` बढ़ाकर
   दोबारा चलाएं ताकि model और अच्छा बने।
6. Training खत्म होने पर notebook के **Output** tab से `checkpoints/best_model.pth`
   डाउनलोड करें।

### 2. App में असली model जोड़ें
```bash
# आपके "app" folder के अंदर:
mkdir -p models/weights
# best_model.pth को models/weights/ में डालें

pip install torch torchvision timm facenet-pytorch opencv-python-headless

# original dummy detector की जगह real version रखें:
cp image_detector_real.py models/image_detector.py
```

### 3. App फिर से चलाएं
```bash
python app.py
```
अब जब भी कोई image upload होगी, असली trained model predict करेगा। अगर किसी वजह से
weights file नहीं मिली, तो app अपने आप पुराने dummy logic पर वापस चला जाएगा — app कभी
crash नहीं होगा।

## Video और Audio के लिए

`video_detector.py` में वही trained face-classifier इस्तेमाल हो सकता है — कई frames पर
चलाकर उनके fake-probability का average/max लेकर एक video-level score बनाया जा सकता है
(यह pattern पहले से `video_detector.py` में TODO comments के रूप में लिखा हुआ है)।

Audio (voice) deepfake detection एक अलग तरह का model चाहता है (spectral/MFCC-based,
video की तरह face-detection based नहीं) — इसके लिए अलग से training pipeline चाहिए होगी
(जैसे ASVspoof dataset पर), जो चाहें तो मैं अगली बार बना सकता हूं।

## ज़रूरी बातें

- असली deepfake detection models को अच्छे results के लिए **बहुत सारा data + GPU समय**
  चाहिए होता है — यह notebook एक ठोस शुरुआत देता है, production-grade accuracy के लिए
  आपको ज़्यादा epochs, ज़्यादा videos, और शायद ensemble models चाहिए होंगे।
- Dataset licensing ज़रूर पढ़ें — FaceForensics++ और DFDC दोनों academic/research use
  के लिए हैं, इस्तेमाल से पहले उनकी terms accept करनी होती हैं।
