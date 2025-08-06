# Model Configuration
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# File Paths
TRAIN_DATA_PATH = "../data/train.csv"
RESULTS_DIR = "final_results"
EVALUATIONS_FILE = "final_results/evaluations.txt"

# Processing Parameters
MIN_THRESHOLD = 10  #minimum samples required for a category before added to other category
BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 5000

# Text Normalization - Noise Words
NOISE_WORDS = {
    # English noise words
    'gram', 'gm', 'ml', 'pieces', 'kg', 'g', 'pcs', 'liter', 'l', 'pack', 
    'bottle', 'can', 'box', 'piece', 'jar', 'bag', 'carton', 'packet', 
    'case', 'each', 'per', 'unit', 'weight', 'x', 'with', 'natural', 
    'pure', 'fresh', 'hot', 'small', 'mix',
    # Arabic noise words
    'جم', 'مل', 'لتر', 'جرام', 'كجم', 'ق', 'ك', 'م', 'قطع', 'قطعة', 'ج',
    'علبه', 'كانز', 'بلاستيك', 'زجاج', 'زجاجه', 'عبوة', 'باكو', 'كيس',
    'وزن', 'من', 'ساده', 'حار', 'فريش', 'طبيعى', 'طبيعي', 'صغير', 'كبير', 
    'جديد', 'ميكس'
}

# Category Preprocessing - Merge Mapping
# Key is the preferred category name, values are variations to merge
CATEGORY_MERGE_MAP = {
    'vegetables and fruits': [
        'vegetables and fruits', 
        'fruits', 
        'vegetables and herbs',
    ],
    'tea coffee and hot drinks': [
        'tea coffee and hot drinks',
        'tea and coffee',
    ],
    'sauces dressings and condiments': [
        'sauces dressings and condiments',
        'condiments dressings and marinades',
    ],
    'chocolates sweets and desserts': [
        'chocolates sweets and desserts',
        'sweets and desserts',
    ],
    'beef and meat': [
        'beef and processed meat',
        'beef and lamb meat',
    ],
    'personal care and body care': [
        'personal care skin and body care',
        'hair shower bath and soap',
    ],
}

# Text Processing Configuration
TEXT_NORMALIZATION_CONFIG = {
    'remove_numbers': True,
    'lowercase': True,
    'remove_special_chars': True,
    'min_word_length': 2,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'f1_average': 'weighted',
    'zero_division': 0,
    'dummy_strategy': 'most_frequent',
}