import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model.eval()

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    #lowercase and remove numbers
    text = re.sub(r'\d+', '', str(text).lower())
    
    #delete special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    noise_words = {
        # English noise words
        'gram', 'gm', 'ml', 'pieces', 'kg', 'g', 'pcs', 'liter', 'l', 'pack', 'bottle', 'can', 'box', 'piece', 'jar', 'bag', 'carton', 'packet', 'case', 'each', 'per', 'unit', 'weight', 'x', 'with', 'natural', 'pure', 'fresh', 'hot', 'small', 'mix',
        # Arabic noise words
        'جم', 'مل', 'لتر', 'جرام', 'كجم', 'ق', 'ك', 'م', 'قطع', 'قطعة', 'ج',
        'علبه', 'كانز', 'بلاستيك', 'زجاج', 'زجاجه', 'عبوة', 'باكو', 'كيس',
        'وزن', 'من', 'ساده', 'حار', 'فريش', 'طبيعى', 'طبيعي', 'صغير', 'كبير', 'جديد', 'ميكس'
    }
    words = [w for w in text.split() if w not in noise_words and len(w) > 1]
    
    #remove extra whitespace
    return ' '.join(words).strip()

def apply_minimum_threshold(df, min_threshold=10):
    """Move categories below threshold to 'Other' category"""
    class_counts = df['class'].value_counts()
    rare_categories = class_counts[class_counts < min_threshold].index.tolist()
    
    df_copy = df.copy()
    df_copy.loc[df_copy['class'].isin(rare_categories), 'class'] = 'Other'
    
    print(f"Moved {len(rare_categories)} categories with <{min_threshold} samples to 'Other'")
    print(f"Affected categories: {rare_categories}")
    
    return df_copy

def preprocess_categories(categories, return_mapping=False):
    #first we Normalize (lowercase, replace '&' with 'and', remove commas, strip)
    normalized = {}
    for cat in categories:
        if pd.notna(cat) and isinstance(cat, str):
            norm_cat = cat.lower().replace('&', 'and').replace(',', '').strip()
            norm_cat = ' '.join(norm_cat.split())
            normalized[cat] = norm_cat
    
    corrected = {}
    for orig_cat, norm_cat in normalized.items():
        corrected_cat = norm_cat.replace('stationary', 'stationery')
        corrected[orig_cat] = corrected_cat
    
    #Key is the preferred category name, values are variations to merge
    merge_map = {
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
    
    #create a reverse mapping (key is each variation, value is the category it's mapped to)
    item_to_category = {}
    for main_cat, variations in merge_map.items():
        for variation in variations:
            item_to_category[variation] = main_cat
    
    #apply merges
    original_to_preprocessed = {}
    final_categories = set()
    
    for orig_cat, corrected_cat in corrected.items():
        if corrected_cat in item_to_category:
            #map to the main category
            final_cat = item_to_category[corrected_cat]
        else:
            #keep as is if not in merge map
            final_cat = corrected_cat
        
        original_to_preprocessed[orig_cat] = final_cat
        final_categories.add(final_cat)
    
    #sort for consistency and convert to list
    cleaned_list = sorted(list(final_categories))
    
    if return_mapping:
        return cleaned_list, original_to_preprocessed
    else:
        return cleaned_list

def get_embeddings(texts):
    clean_texts = [normalize_text(text) for text in texts]
    embeddings = model.encode(clean_texts)
    return embeddings

def predict_category(item_embedding, category_embeddings, categories):
    similarities = cosine_similarity([item_embedding], category_embeddings)[0]
    best_idx = np.argmax(similarities)
    return categories[best_idx]

def main():
    print("Loading data...")
    df = pd.read_csv("train.csv")
    
    print("Applying minimum sample threshold...")
    df = apply_minimum_threshold(df, min_threshold=10)
    
    original_categories = [cat for cat in df['class'].unique() if pd.notna(cat) and isinstance(cat, str)]
    print(f"Found {len(original_categories)} unique categories before preprocessing")
    
    print("Preprocessing categories...")
    all_categories, category_mapping = preprocess_categories(original_categories, return_mapping=True)
    print(f"After preprocessing: {len(all_categories)} categories")
    
    df_filtered = df.copy()
    df_filtered['preprocessed_class'] = df_filtered['class'].map(category_mapping).fillna(df_filtered['class'])
    print(f"Using all {len(all_categories)} categories with {len(df_filtered)} items (full dataset)")
    
    os.makedirs("final_results", exist_ok=True)
    if os.path.exists("final_results/evaluations.txt"):
        os.remove("final_results/evaluations.txt")
    
    print("Generating category embeddings...")
    category_embeddings = get_embeddings(all_categories)
    
    batch_size = 1000
    results = []
    total_items = len(df_filtered)
    
    print(f"Processing {total_items} items...")
    
    for i in range(0, total_items, batch_size):
        batch_df = df_filtered.iloc[i:i+batch_size]
        item_names = batch_df['Item_Name'].tolist()
        true_categories = batch_df['preprocessed_class'].tolist()
        original_categories = batch_df['class'].tolist()
        
        item_embeddings = get_embeddings(item_names)
        
        #predict categories for batch
        for j, item_embedding in enumerate(item_embeddings):
            predicted_category = predict_category(item_embedding, category_embeddings, all_categories)
            
            results.append({
                'input_item_name': item_names[j],
                'predicted_category': predicted_category,
                'true_category': true_categories[j],
                'original_category': original_categories[j]
            })
        
        print(f"Processed {min(i + batch_size, total_items)}/{total_items} items")
        
        #save intermediate results every 5000 items
        if (i + batch_size) % 5000 == 0 or i + batch_size >= total_items:
            checkpoint_number = str(((i + batch_size) // 5000)).zfill(3)
            
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"final_results/checkpoint_{checkpoint_number}.csv", index=False)
            
            y_true_checkpoint = temp_df['true_category'].astype(str)
            y_pred_checkpoint = temp_df['predicted_category'].astype(str)
            
            #dummy classifier
            dummy_clf = DummyClassifier(strategy="most_frequent")
            dummy_clf.fit(y_true_checkpoint.values.reshape(-1, 1), y_true_checkpoint)
            y_dummy_checkpoint = dummy_clf.predict(y_true_checkpoint.values.reshape(-1, 1))
            
            f1_model_checkpoint = f1_score(y_true_checkpoint, y_pred_checkpoint, average='weighted', zero_division=0)
            f1_dummy_checkpoint = f1_score(y_true_checkpoint, y_dummy_checkpoint, average='weighted', zero_division=0)
            improvement = f1_model_checkpoint - f1_dummy_checkpoint
            
            # Append to evaluation file
            with open("final_results/evaluations.txt", "a") as f:
                if i == 0:  # First checkpoint, write header
                    f.write("Checkpoint Evaluations - Embedding-based Classification\n")
                    f.write("========================================================\n\n")
                f.write(f"Checkpoint {checkpoint_number} ({i+batch_size} items):\n")
                f.write(f"  Model F1 Score: {f1_model_checkpoint:.3f}\n")
                f.write(f"  Dummy F1 Score: {f1_dummy_checkpoint:.3f}\n")
                f.write(f"  Improvement: {improvement:.3f}\n\n")
            
            print(f"Checkpoint {checkpoint_number}: F1={f1_model_checkpoint:.3f}, Dummy={f1_dummy_checkpoint:.3f}, Improvement={improvement:.3f}")
            print(f"Saved checkpoint at {i+batch_size} items")
    
    # Evaluate results
    results_df = pd.DataFrame(results)
    y_true = results_df['true_category'].astype(str)
    y_pred = results_df['predicted_category'].astype(str)
    
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(y_true.values.reshape(-1, 1), y_true)
    y_dummy = dummy_clf.predict(y_true.values.reshape(-1, 1))
    
    f1_model = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_dummy = f1_score(y_true, y_dummy, average='weighted', zero_division=0)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Model F1 Score: {f1_model:.3f}")
    print(f"Dummy F1 Score: {f1_dummy:.3f}")
    print(f"Improvement: {f1_model - f1_dummy:.3f}")
    
    # Save final results
    results_df.to_csv("final_results/embedding_predictions.csv", index=False)
    
    # Append final summary to evaluations file
    with open("final_results/evaluations.txt", "a") as f:
        f.write("="*60 + "\n")
        f.write("FINAL RESULTS - FULL DATASET\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total items processed: {len(results_df)}\n")
        f.write(f"Total categories: {len(all_categories)}\n\n")
        f.write(f"Final Model F1 Score: {f1_model:.3f}\n")
        f.write(f"Final Dummy F1 Score: {f1_dummy:.3f}\n")
        f.write(f"Final Improvement: {f1_model - f1_dummy:.3f}\n\n")
        f.write("Categories used:\n")
        # Filter out any NaN values and sort the categories
        valid_categories = [cat for cat in all_categories if pd.notna(cat) and isinstance(cat, str)]
        for i, cat in enumerate(sorted(valid_categories), 1):
            f.write(f"{i:2d}. {cat}\n")
    
    print(f"\nFinal results saved to final_results/")


main()