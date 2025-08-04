import csv
import json
import re
import pandas as pd
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from llama_cpp import Llama

print("Loading Phi model...")
llm = Llama.from_pretrained(
    repo_id="TheBloke/phi-2-GGUF",
    filename="phi-2.Q4_K_M.gguf",
    verbose=False
)

def predict_category(item_name):
    """Predict category from item name using LLM"""
    prompt = f"""Predict the category for this item. Return only the category name.

Item: {item_name}

Category:"""
    
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30, 
        temperature=0.2
    )
    
    predicted_category = response['choices'][0]['message']['content'].strip()
    return predicted_category

def evaluate_predictions(results_df):
    y_true = results_df['true_category'].astype(str)
    y_pred = results_df['predicted_category'].astype(str)
    
    #dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(y_true.values.reshape(-1, 1), y_true)
    y_dummy = dummy_clf.predict(y_true.values.reshape(-1, 1))
    
    f1_model = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_dummy = f1_score(y_true, y_dummy, average='weighted', zero_division=0)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Model F1 Score: {f1_model:.3f}")
    print(f"Dummy F1 Score: {f1_dummy:.3f}")
    print(f"Improvement: {f1_model - f1_dummy:.3f}")
    
    return f1_model, f1_dummy

def main():
    print("Processing...")
    
    os.makedirs("final_results", exist_ok=True)
    with open("final_results/evaluations.txt", "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")
    checkpoint_counter = 0

    df = pd.read_csv("train.csv")
    total_records = len(df)
    print(f"Processing {total_records} records from dataset...")
    
    results = []
    
    for idx, row in df.iterrows():
        item_name = row['Item_Name']
        
        print(f"\n--- Record {idx + 1}/{total_records} ---")
        print(f"Item: {item_name}")
        
        #use llm
        predicted_category = predict_category(item_name)
        
        #store results
        result = {
            'input_item_name': item_name,
            'predicted_category': predicted_category,
            'true_category': row['class']
        }
        results.append(result)
        
        print(f"Predicted Category: {predicted_category}")
        
        if (idx + 1) % 50 == 0:
            checkpoint_counter += 1
            checkpoint_df = pd.DataFrame(results[-50:])  # Last 50 records
            checkpoint_filename = f"final_results/checkpoint_{checkpoint_counter:03d}.csv"
            checkpoint_df.to_csv(checkpoint_filename, index=False, encoding='utf-8')
            print(f"Checkpoint saved: {checkpoint_filename}")

            checkpoint_results_df = pd.DataFrame(results)
            f1_model, f1_dummy = evaluate_predictions(checkpoint_results_df)
            
            #save f1 scores to txt file
            with open("final_results/evaluations.txt", "a") as f:
                f.write(f"Checkpoint {checkpoint_counter} (Records: {idx + 1})\n")
                f.write(f"Model F1: {f1_model:.3f}, Dummy F1: {f1_dummy:.3f}, Improvement: {f1_model - f1_dummy:.3f}\n\n")  
        
    #save final results to csv
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("final_results/category_predictions_final.csv", index=False, encoding='utf-8')
        print(f"\nProcessed {len(results)} total records. Results saved to final_results/")
        evaluate_predictions(results_df)

main()