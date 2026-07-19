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
echo "   (main.py akan menjalankan 5 notebook secara berurutan)"
echo "------------------------------------------------------------"

# Jalankan dari root project agar path relatif benar
python src/main.py

echo ""
echo "============================================================"
echo "  Pipeline selesai. Cek folder output/ untuk hasil."
echo "============================================================"
