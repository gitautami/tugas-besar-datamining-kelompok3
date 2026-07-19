#!/bin/bash

echo "============================================================"
echo "  PIPELINE SENTIMEN ANALISIS DATA MINING – Kelompok 3"
echo "============================================================"

# Optional: Cek dan install requirements
if [ -f requirements.txt ]; then
    echo ""
    echo "📦 Menginstall dependensi dari requirements.txt..."
    pip install -r requirements.txt -q
    echo "✅ Dependensi siap."
fi

echo ""
echo "🚀 Menjalankan pipeline: src/main.py"
echo "------------------------------------------------------------"

# Jalankan dari root project agar path relatif benar
python3 src/main.py

echo ""
echo "============================================================"
echo "  Pipeline selesai. Cek folder output/ untuk hasil."
echo "============================================================"
