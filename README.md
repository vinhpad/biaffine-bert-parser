# Biaffine BERT Dependency Parser for Vietnamese

Triển khai mô hình **Biaffine Dependency Parser** với BERT backbone cho phân tích cú pháp phụ thuộc tiếng Việt theo chuẩn Universal Dependencies.

## Tổng quan

Mô hình này kết hợp:
- **BERT encoder**: Sử dụng PhoBERT (vinai/phobert-base) để encode ngữ cảnh
- **Biaffine Attention**: Tính điểm tương tác giữa head và dependent 
- **MST Decoding**: Thuật toán Chu-Liu/Edmonds để tìm cây phụ thuộc tối ưu

## Kiến trúc

```
Input: "Tôi thích học tiếng Việt"
   ↓
BERT Encoder (PhoBERT)
   ↓
Word-level Representations
   ↓
Arc MLP + Label MLP
   ↓
Biaffine Attention
   ↓
Arc Scores + Label Scores  
   ↓
MST Decoder
   ↓
Dependency Tree
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu:
```bash
# Copy dữ liệu UD Vietnamese vào thư mục data/
cp ../vi_vtb-ud-*.conllu data/
```

## Sử dụng

### Training

```bash
cd src/training
python train.py --train_path ../../data/vi_vtb-ud-train.conllu \
                --dev_path ../../data/vi_vtb-ud-dev.conllu \
                --test_path ../../data/vi_vtb-ud-test.conllu \
                --output_dir ../../experiments/biaffine_parser \
                --batch_size 16 \
                --max_epochs 50
```

### Với config file:

```bash
python train.py --config config/default.yaml
```

## Cấu trúc Project

```
biaffine-bert-parser/
├── src/
│   ├── models/
│   │   ├── bert_encoder.py       # BERT encoder wrapper
│   │   └── biaffine_parser.py    # Main parser model
│   ├── modules/
│   │   └── biaffine_attention.py # Biaffine attention implementation
│   ├── datasets/
│   │   └── ud_dataset.py         # CoNLL-U dataset loader
│   ├── training/
│   │   └── train.py              # Training script
│   └── utils/
│       └── mst_decoder.py        # MST decoding algorithm
├── data/                         # Dataset files
├── experiments/                  # Training outputs
├── requirements.txt
└── README.md
```

## Thuật toán MST

Mô hình sử dụng **Chu-Liu/Edmonds' algorithm** để đảm bảo output là cây dependency hợp lệ:

1. **Step 1**: Với mỗi từ, chọn head có điểm cao nhất
2. **Step 2**: Kiểm tra cycles 
3. **Step 3**: Nếu có cycles, contract các cycles và giải đệ quy
4. **Step 4**: Expand solution để có cây cuối cùng

## Metrics

- **UAS (Unlabeled Attachment Score)**: Tỷ lệ từ có head đúng
- **LAS (Labeled Attachment Score)**: Tỷ lệ từ có cả head và label đúng

## Kết quả mong đợi

Với dataset UD Vietnamese-VTB:
- UAS: ~85-90%
- LAS: ~80-85%

## Tham số mô hình

- **BERT**: vinai/phobert-base (768 hidden size)
- **Arc MLP**: 500 hidden units
- **Label MLP**: 100 hidden units  
- **Dropout**: 0.33
- **Learning rate**: 2e-5 (BERT), 1e-4 (other)
- **Batch size**: 16
- **Max epochs**: 50

## Tham khảo

- Dozat & Manning (2017). "Deep Biaffine Attention for Neural Dependency Parsing"
- Nguyen et al. (2020). "PhoBERT: Pre-trained language models for Vietnamese"
- Universal Dependencies Vietnamese-VTB treebank