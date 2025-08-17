import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from datetime import datetime
import openai
from openai import OpenAI
import json
import openpyxl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

    

def load_and_clean_data(file_path: str) -> pd.DataFrame:

    """
    Load the feedback file (Excel or CSV) and clean the data
    """
    import os
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv']:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Remove rows with blank answers
    df = df.dropna(subset=['Answer'])

    # Remove duplicate responses
    df = df.drop_duplicates(subset=['Employee ID', 'Question', 'Answer'])

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    return df

def get_embeddings(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI's API"""
    embeddings = []
    for i, text in enumerate(texts):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append(response.data[0].embedding)
            if (i + 1) % 10 == 0:
                print(f"Generated embeddings for {i + 1}/{len(texts)} texts")
        except Exception as e:
            print(f"Error getting embedding for text {i}: {str(e)}")
            # Use zero vector as fallback
            embeddings.append(np.zeros(1536))
    return np.array(embeddings)

def cluster_feedback(embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 10, target_variance: float = 0.95) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster feedback using PCA and KMeans
    
    Args:
        embeddings: Input embeddings matrix
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        target_variance: Target explained variance ratio (0.0 to 1.0, default: 0.9)
                       Higher values preserve more information but use more dimensions
    
    Returns:
        Tuple of cluster centers and cluster assignments
    """
    # First fit PCA with maximum possible components
    max_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(embeddings)
    
    # Find number of components needed to explain target variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= target_variance) + 1
    print(f"Using {n_components} components to explain {target_variance*100:.1f}% of variance")
    
    # Apply PCA with selected number of components
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Calculate optimal number of clusters between min_clusters and max_clusters
    n_clusters = max(min_clusters, min(max_clusters, len(embeddings) // 5))
    
    # Cluster the reduced embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(reduced_embeddings)
    
    # Get cluster centers in original space
    cluster_centers = []
    for i in range(n_clusters):
        cluster_points = embeddings[cluster_assignments == i]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        else:
            # Fallback if cluster is empty
            cluster_centers.append(np.zeros_like(embeddings[0]))
    
    return np.array(cluster_centers), cluster_assignments

def analyze_themes(
    df: pd.DataFrame,
    use_clustering: bool = True,
    gpt_model: str = 'gpt-4',
    assignment_model: str = 'gpt-3.5-turbo',
    max_clusters: int = 10
) -> pd.DataFrame:
    """
    Perform thematic analysis using either:
    1. Embedding-based clustering with PCA and K-means
    2. Direct GPT-based theme identification

    Args:
        df: Input DataFrame with feedback
        use_clustering: Whether to use clustering approach
        gpt_model: GPT model to use for theme exploration ('gpt-3.5-turbo', 'gpt-4', or 'gpt-4-turbo-preview')
        assignment_model: GPT model to use for theme assignment (default: 'gpt-3.5-turbo')
    """
    # Initialize OpenAI client
    client = OpenAI()  # Make sure to set OPENAI_API_KEY in your environment variables
    
    # Create empty columns for themes and sentiment
    df['Main Theme'] = ''
    df['Sub-theme'] = ''
    df['Cluster'] = -1
    df['Overall_Sentiment'] = 0.0
    df['Engagement_Level'] = 0.0
    df['Constructiveness'] = 0.0
    
    
    if use_clustering:
        try:
            print("Step 1: Generating embeddings...")
            embeddings = get_embeddings(client, df['Answer'].tolist())

            print(f"\nStep 2: Clustering feedback for main themes (PCA variance=95.0%, max_clusters={max_clusters})...")
            cluster_centers, cluster_assignments = cluster_feedback(embeddings, max_clusters=max_clusters, target_variance=0.95)
            df['Cluster'] = cluster_assignments

            # Sentiment analysis for each main theme (after clusters assigned)
            print("Analyzing sentiment for each main theme (up to 10 samples per theme)...")
            for cluster_id in range(len(cluster_centers)):
                mask = df['Cluster'] == cluster_id
                cluster_df = df[mask]
                if len(cluster_df) == 0:
                    continue
                sample_answers = cluster_df['Answer'].tolist()
                sample_count = min(10, len(sample_answers))
                sample_indices = np.random.choice(len(sample_answers), size=sample_count, replace=False)
                batch_texts = [sample_answers[i] for i in sample_indices]
                # Build a single prompt for all samples
                batch_prompt = """
You are a sentiment analysis expert. For each feedback below, analyze and respond with a JSON object containing exactly these three numeric scores (from -1.0 to 1.0):
- overall_sentiment: general sentiment (-1.0 most negative to 1.0 most positive)
- engagement_level: how engaged/passionate (-1.0 completely disengaged to 1.0 highly engaged)
- constructiveness: how constructive/actionable (-1.0 not constructive to 1.0 highly constructive)

Feedbacks:
"""
                for idx, text in enumerate(batch_texts):
                    batch_prompt += f"\n{idx+1}. \"{text}\""
                batch_prompt += """

Respond with ONLY a JSON array of objects in this exact format (no other text):
[
  {"overall_sentiment": number, "engagement_level": number, "constructiveness": number},
  ...
]
"""
                sentiments = []
                try:
                    response = client.chat.completions.create(
                        model=assignment_model,
                        messages=[
                            {"role": "system", "content": "You are a sentiment analysis expert. Respond ONLY with valid JSON arrays."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        temperature=0.2
                    )
                    content = response.choices[0].message.content.strip()
                    # Try to extract JSON array if there's extra text
                    if not (content.startswith('[') and content.endswith(']')):
                        json_start = content.find('[')
                        json_end = content.rfind(']')
                        if json_start >= 0 and json_end > json_start:
                            content = content[json_start:json_end+1]
                    results = json.loads(content)
                    for result in results:
                        sentiments.append([
                            float(result.get('overall_sentiment', 0.0)),
                            float(result.get('engagement_level', 0.0)),
                            float(result.get('constructiveness', 0.0))
                        ])
                    print(f"Cluster {cluster_id} sample sentiments: {results}")
                except Exception as e:
                    print(f"Error analyzing sentiment for cluster {cluster_id} batch: {str(e)}")
                    # Fill with zeros if error
                    for _ in range(sample_count):
                        sentiments.append([0.0, 0.0, 0.0])
                if sentiments:
                    avg_sentiment = np.mean(sentiments, axis=0)
                else:
                    avg_sentiment = [0.0, 0.0, 0.0]
                df.loc[mask, 'Overall_Sentiment'] = avg_sentiment[0]
                df.loc[mask, 'Engagement_Level'] = avg_sentiment[1]
                df.loc[mask, 'Constructiveness'] = avg_sentiment[2]

            # Name main themes using GPT
            main_theme_names = {}
            for cluster_id in range(len(cluster_centers)):
                cluster_texts = df[df['Cluster'] == cluster_id]['Answer'].tolist()
                if not cluster_texts:
                    continue
                samples = np.random.choice(cluster_texts, size=min(3, len(cluster_texts)), replace=False)
                theme_prompt = f"""
                Analyze these similar feedback entries and identify:
                1. A main theme that captures their primary topic/concern (1-4 words)

                Feedback entries:
                {chr(10).join(f"- {text}" for text in samples)}

                You must respond with valid JSON in exactly this format (no other text):
                {{
                    "main_theme": "theme name"
                }}
                """
                try:
                    theme_response = client.chat.completions.create(
                        model=assignment_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at identifying themes in qualitative feedback."},
                            {"role": "user", "content": theme_prompt}
                        ],
                        temperature=0.3
                    )
                    content = theme_response.choices[0].message.content.strip()
                    result = json.loads(content)
                    main_theme = result.get('main_theme', f"Main Theme {cluster_id+1}")
                    main_theme_names[cluster_id] = main_theme
                    df.loc[df['Cluster'] == cluster_id, 'Main Theme'] = main_theme
                    print(f"Cluster {cluster_id}: {main_theme}")
                except Exception as e:
                    print(f"Error naming main theme for cluster {cluster_id}: {str(e)}")
                    main_theme_names[cluster_id] = f"Main Theme {cluster_id+1}"
                    df.loc[df['Cluster'] == cluster_id, 'Main Theme'] = f"Main Theme {cluster_id+1}"

            # Sub-theme clustering within each main theme
            print("\nStep 3: Sub-theme clustering within each main theme (PCA variance=95.0%)...")
            for cluster_id in range(len(cluster_centers)):
                mask = df['Cluster'] == cluster_id
                sub_df = df[mask]
                if len(sub_df) < 2:
                    # Not enough data to sub-cluster
                    df.loc[mask, 'Sub-theme'] = 'Miscellaneous'
                    continue
                # Get embeddings for this cluster
                sub_embeddings = get_embeddings(client, sub_df['Answer'].tolist())
                try:
                    # Dynamically determine n_clusters for sub-themes (2 to min(6, len(sub_df)))
                    from sklearn.metrics import silhouette_score
                    best_n = 2
                    best_score = -1
                    max_sub_clusters = min(6, len(sub_df))
                    for n in range(2, max_sub_clusters+1):
                        try:
                            pca = PCA(n_components=min(sub_embeddings.shape[0], sub_embeddings.shape[1]))
                            pca.fit(sub_embeddings)
                            cumsum = np.cumsum(pca.explained_variance_ratio_)
                            n_components = np.argmax(cumsum >= 0.95) + 1
                            reduced = PCA(n_components=n_components).fit_transform(sub_embeddings)
                            kmeans = KMeans(n_clusters=n, random_state=42)
                            labels = kmeans.fit_predict(reduced)
                            score = silhouette_score(reduced, labels) if n < len(sub_df) else -1
                            if score > best_score:
                                best_score = score
                                best_n = n
                        except Exception:
                            continue
                    # Final sub-theme clustering
                    pca = PCA(n_components=min(sub_embeddings.shape[0], sub_embeddings.shape[1]))
                    pca.fit(sub_embeddings)
                    cumsum = np.cumsum(pca.explained_variance_ratio_)
                    n_components = np.argmax(cumsum >= 0.95) + 1
                    reduced = PCA(n_components=n_components).fit_transform(sub_embeddings)
                    kmeans = KMeans(n_clusters=best_n, random_state=42)
                    sub_labels = kmeans.fit_predict(reduced)
                    # Assign sub-theme cluster numbers
                    df.loc[mask, 'Sub-theme'] = [f"Sub-theme {i+1}" for i in sub_labels]
                    # Name sub-themes using GPT
                    for sub_id in range(best_n):
                        sub_mask = (df['Cluster'] == cluster_id) & (df['Sub-theme'] == f"Sub-theme {sub_id+1}")
                        sub_texts = df[sub_mask]['Answer'].tolist()
                        if not sub_texts:
                            continue
                        sub_samples = np.random.choice(sub_texts, size=min(3, len(sub_texts)), replace=False)
                        sub_prompt = f"""
                        The main theme is: '{main_theme_names[cluster_id]}'
                        Analyze these feedback entries and suggest a specific sub-theme (2-6 words) that best describes their shared aspect.

                        Feedback entries:
                        {chr(10).join(f"- {text}" for text in sub_samples)}

                        Respond with valid JSON in this format (no other text):
                        {{
                            "sub_theme": "sub-theme name"
                        }}
                        """
                        try:
                            sub_response = client.chat.completions.create(
                                model=assignment_model,
                                messages=[
                                    {"role": "system", "content": "You are an expert at identifying sub-themes in qualitative feedback."},
                                    {"role": "user", "content": sub_prompt}
                                ],
                                temperature=0.3
                            )
                            sub_content = sub_response.choices[0].message.content.strip()
                            sub_result = json.loads(sub_content)
                            sub_theme_name = sub_result.get('sub_theme', f"Sub-theme {sub_id+1}")
                            df.loc[sub_mask, 'Sub-theme'] = sub_theme_name
                            print(f"  Named sub-theme {sub_id+1} for main theme '{main_theme_names[cluster_id]}': {sub_theme_name}")
                        except Exception as e:
                            print(f"  Error naming sub-theme {sub_id+1} for main theme '{main_theme_names[cluster_id]}': {str(e)}")
                            # Keep default name
                except Exception as e:
                    print(f"Error in sub-theme clustering for main theme {main_theme_names[cluster_id]}: {str(e)}")
                    df.loc[mask, 'Sub-theme'] = 'Miscellaneous'
        except Exception as e:
            print(f"Error in hierarchical clustering approach: {str(e)}")
            use_clustering = False
    

    if not use_clustering:
        # Optimized GPT-based theme identification with batching
        print("Step 1: Exploring themes across all feedback...")

        # Step 1: Theme exploration (batched for large datasets)
        print("\nStep 1: Exploring themes across all feedback in batches...")
        batch_size = 100  # Adjust as needed for token limits
        n = len(df)
        all_themes = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = df.iloc[start:end]
            numbered_feedback = "\n".join([f"Entry {i+1}: {row['Answer']}" for i, row in batch.iterrows()])
            explore_prompt = f"""
You are conducting a thematic analysis of employee feedback. 
Read the following feedback entries carefully and:
1. Identify recurring patterns, sentiments, and topics
2. Create a structured list of main themes (1 to 4 words) and their sub-themes (2 to 6 words)
3. Each main theme should capture a key aspect of the feedback
4. Sub-themes should represent specific aspects of each main theme

Feedback entries:
{numbered_feedback}

Provide the response in JSON format ONLY (no additional text):
{{
    "themes": [
        {{
            "main_theme": "Theme Name",
            "sub_themes": [
                "Sub-theme name 1",
                "Sub-theme name 2"
            ]
        }}
    ]
}}
"""
            max_attempts = 3
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    print(f"\nTheme exploration batch {start+1}-{end} (attempt {attempt})...")
                    explore_response = client.chat.completions.create(
                        model=gpt_model,
                        messages=[
                            {"role": "system", "content": "You are an expert in qualitative thematic analysis. Respond ONLY with valid JSON."},
                            {"role": "user", "content": explore_prompt}
                        ],
                        temperature=0.2
                    )
                    raw_content = explore_response.choices[0].message.content.strip()
                    print(f"\nRaw theme exploration response (first 200 chars): {raw_content[:200]}...")
                    json_content = raw_content
                    if not (raw_content.startswith('{') and raw_content.endswith('}')):
                        json_start = raw_content.find('{')
                        json_end = raw_content.rfind('}')
                        if json_start >= 0 and json_end > json_start:
                            json_content = raw_content[json_start:json_end+1]
                            print(f"Extracted JSON content from response")
                    themes_structure = json.loads(json_content)
                    if 'themes' not in themes_structure:
                        print("Missing 'themes' key in response, retrying...")
                        continue
                    if not all('main_theme' in theme and 'sub_themes' in theme for theme in themes_structure['themes']):
                        print("Invalid theme structure, missing required keys, retrying...")
                        continue
                    all_themes.extend(themes_structure['themes'])
                    break
                except Exception as e:
                    print(f"Error in theme exploration batch {start+1}-{end} attempt {attempt}: {str(e)}")
                    if attempt >= max_attempts:
                        print("All theme exploration attempts failed for this batch.")
                        raise

        # Merge all themes (deduplicate by main_theme and sub_theme)
        merged_themes = {}
        for theme in all_themes:
            main = theme['main_theme']
            subs = set(theme.get('sub_themes', []))
            if main in merged_themes:
                merged_themes[main].update(subs)
            else:
                merged_themes[main] = set(subs)
        themes_structure = {
            'themes': [
                {'main_theme': main, 'sub_themes': sorted(list(subs))}
                for main, subs in merged_themes.items()
            ]
        }
        success = len(themes_structure['themes']) > 0
        print("Identified merged themes structure:")
        for theme in themes_structure['themes']:
            print(f"\nMain Theme: {theme['main_theme']}")
            for sub_theme in theme['sub_themes']:
                print(f"  - Sub-theme: {sub_theme}")

        # Step 2: Assign sentiment and themes in batches
        if success and themes_structure:
            try:
                print("\nStep 2: Assigning sentiment and themes to feedback in batches of 5...")
                batch_size = 5
                n = len(df)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch = df.iloc[start:end]
                    numbered_feedback = "\n".join([f"Entry {i+1}: {row['Answer']}" for i, row in batch.iterrows()])
                    batch_prompt = f"""
Using the following theme structure:
{json.dumps(themes_structure, indent=2)}

Analyze the following feedback entries. For each entry, return a JSON object with these fields:
- overall_sentiment (from -1.0 to 1.0)
- engagement_level (from -1.0 to 1.0)
- constructiveness (from -1.0 to 1.0)
- main_theme (selected main theme)
- sub_theme (selected sub-theme)

Feedback entries:
{numbered_feedback}

Respond with ONLY a JSON array in this format (one object per entry, in order):
[
  {{
    "overall_sentiment": number,
    "engagement_level": number,
    "constructiveness": number,
    "main_theme": "selected main theme",
    "sub_theme": "selected sub-theme"
  }}
]
"""
                    try:
                        batch_response = client.chat.completions.create(
                            model=assignment_model,
                            messages=[
                                {"role": "system", "content": "You are an expert in qualitative thematic analysis and sentiment analysis. Respond ONLY with valid JSON."},
                                {"role": "user", "content": batch_prompt}
                            ],
                            temperature=0.3
                        )
                        content = batch_response.choices[0].message.content.strip()
                        # Try to extract JSON array if there's extra text
                        if not (content.startswith('[') and content.endswith(']')):
                            json_start = content.find('[')
                            json_end = content.rfind(']')
                            if json_start >= 0 and json_end > json_start:
                                content = content[json_start:json_end+1]
                        results = json.loads(content)
                        # Check if results length matches batch size
                        expected = end - start
                        if len(results) < expected:
                            print(f"WARNING: Only {len(results)} results returned for batch {start+1}-{end} (expected {expected}). Filling missing entries with error message.")
                        # Assign results back to DataFrame
                        for i in range(expected):
                            idx = start + i
                            if i < len(results):
                                result = results[i]
                                df.at[idx, 'Overall_Sentiment'] = result.get('overall_sentiment', 0.0)
                                df.at[idx, 'Engagement_Level'] = result.get('engagement_level', 0.0)
                                df.at[idx, 'Constructiveness'] = result.get('constructiveness', 0.0)
                                df.at[idx, 'Main Theme'] = result.get('main_theme', 'Uncategorized')
                                df.at[idx, 'Sub-theme'] = result.get('sub_theme', 'Error in processing')
                            else:
                                df.at[idx, 'Overall_Sentiment'] = 0.0
                                df.at[idx, 'Engagement_Level'] = 0.0
                                df.at[idx, 'Constructiveness'] = 0.0
                                df.at[idx, 'Main Theme'] = "Batch incomplete: No result returned"
                                df.at[idx, 'Sub-theme'] = "Batch incomplete: No result returned"
                        print(f"Processed feedback {start+1}-{end} of {n}")
                    except Exception as e:
                        print(f"Error processing batch {start+1}-{end}: {str(e)}")
                        # On error, fill with defaults
                        for i in range(start, end):
                            df.at[i, 'Overall_Sentiment'] = 0.0
                            df.at[i, 'Engagement_Level'] = 0.0
                            df.at[i, 'Constructiveness'] = 0.0
                            df.at[i, 'Main Theme'] = "Uncategorized"
                            df.at[i, 'Sub-theme'] = "Error in processing"
            except Exception as e:
                print(f"Error in batch sentiment/theme assignment phase: {str(e)}")
                raise
    
    return df

def calculate_frequencies(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate frequencies of main themes and sub-themes
    """
    # Main theme frequencies
    main_theme_freq = df['Main Theme'].value_counts().reset_index()
    main_theme_freq.columns = ['Theme', 'Count']
    
    # Sub-theme frequencies grouped by main theme
    sub_theme_freq = df.groupby(['Main Theme', 'Sub-theme']).size().reset_index()
    sub_theme_freq.columns = ['Main Theme', 'Sub-theme', 'Count']
    
    return main_theme_freq, sub_theme_freq

def create_visualizations(main_theme_freq: pd.DataFrame, sub_theme_freq: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Placeholder function that returns an empty dictionary since we're not creating visualizations
    """
    return {}

def generate_report(df: pd.DataFrame, main_theme_freq: pd.DataFrame, 
                   sub_theme_freq: pd.DataFrame, figures: Dict[str, plt.Figure], 
                   output_path: str, gpt_model: str = 'gpt-3.5-turbo'):
    """
    Generate an Excel report with themed feedback, sentiment scores, and an aggregated summary sheet
    """
    import openai
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_path, f'feedback_analysis_{timestamp}.xlsx')

    # Create the output DataFrame with all columns in desired order
    df_output = df[[
        'Employee ID', 'Question', 'Answer', 
        'Main Theme', 'Sub-theme',
        'Overall_Sentiment', 'Engagement_Level', 'Constructiveness'
    ]].copy()

    # Format sentiment scores to 2 decimal places for readability
    df_output['Overall_Sentiment'] = df_output['Overall_Sentiment'].round(2)
    df_output['Engagement_Level'] = df_output['Engagement_Level'].round(2)
    df_output['Constructiveness'] = df_output['Constructiveness'].round(2)

    # Prepare aggregated report for second sheet
    agg_rows = []
    # Main theme aggregation
    for theme in main_theme_freq['Theme']:
        theme_df = df[df['Main Theme'] == theme]
        row = {
            'Main Theme': theme,
            'Sub-theme': '',
            'Count': len(theme_df),
            'Avg Overall_Sentiment': theme_df['Overall_Sentiment'].mean().round(2),
            'Avg Engagement_Level': theme_df['Engagement_Level'].mean().round(2),
            'Avg Constructiveness': theme_df['Constructiveness'].mean().round(2),
        }
        agg_rows.append(row)
        # Sub-theme aggregation
        for sub in theme_df['Sub-theme'].unique():
            sub_df = theme_df[theme_df['Sub-theme'] == sub]
            row = {
                'Main Theme': theme,
                'Sub-theme': sub,
                'Count': len(sub_df),
                'Avg Overall_Sentiment': sub_df['Overall_Sentiment'].mean().round(2),
                'Avg Engagement_Level': sub_df['Engagement_Level'].mean().round(2),
                'Avg Constructiveness': sub_df['Constructiveness'].mean().round(2),
            }
            agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    # Generate short summary for each main theme using OpenAI (ChatGPT)
    summaries = {}
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        client = openai.OpenAI(api_key=openai_api_key)
        for theme in main_theme_freq['Theme']:
            theme_answers = df[df['Main Theme'] == theme]['Answer'].dropna().tolist()
            prompt = f"""
You are an HR analytics expert. Summarize the following employee feedback for the theme '{theme}' in 2-5 sentences, highlighting key points and sentiment:
{theme_answers[:10]}
"""
            try:
                response = client.chat.completions.create(
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": "You are an HR analytics expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.4
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = f"Error generating summary: {e}"
            summaries[theme] = summary
    else:
        for theme in main_theme_freq['Theme']:
            summaries[theme] = "(No API key set. Summary not generated.)"

    agg_df['Theme Summary'] = agg_df.apply(
        lambda row: summaries[row['Main Theme']] if row['Sub-theme'] == '' else '', axis=1
    )

    # Write to Excel with two sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False, sheet_name='Feedback Data')
        agg_df.to_excel(writer, index=False, sheet_name='Aggregated Report')

def main():

    # Get the Excel file path from user input
    file_path = input("Please enter the path to your Excel file: ")

    # Ask for analysis method
    while True:
        method = input("\nChoose analysis method:\n1. Embedding-based clustering (faster for large datasets)\n2. Direct GPT analysis (better for small datasets)\nEnter 1 or 2: ").strip()
        if method in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    use_clustering = (method == '1')

    # PCA variance presets
    presets = {
        '1': ('Less Variance', 0.90, 0.93),
        '2': ('Standard Variance', 0.95, 0.97),
        '3': ('More Variance', 0.99, 0.995)
    }
    print("\nSelect PCA variance preset:")
    for k, v in presets.items():
        print(f"{k}. {v[0]} (Main: {v[1]*100:.1f}%, Sub: {v[2]*100:.1f}%)")
    while True:
        preset_choice = input("Enter 1, 2, or 3 for PCA variance preset: ").strip()
        if preset_choice in presets:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    preset_name, main_var, sub_var = presets[preset_choice]
    print(f"\nUsing preset: {preset_name} (Main: {main_var*100:.1f}%, Sub: {sub_var*100:.1f}%)")

    # Create output directory if it doesn't exist
    output_path = 'analysis_output'
    os.makedirs(output_path, exist_ok=True)

    # Load and process the data
    print("\nLoading and cleaning data...")
    df = load_and_clean_data(file_path)

    print(f"\nAnalyzing {len(df)} feedback entries using {'clustering' if use_clustering else 'direct GPT'} approach...")
    # Perform thematic analysis
    df = analyze_themes(df, use_clustering=use_clustering, main_theme_variance=main_var, sub_theme_variance=sub_var)

    print("\nCalculating theme frequencies...")
    # Calculate frequencies
    main_theme_freq, sub_theme_freq = calculate_frequencies(df)

    # Create visualizations (empty for now)
    figures = create_visualizations(main_theme_freq, sub_theme_freq)

    print("\nGenerating report...")
    # Generate final report
    generate_report(df, main_theme_freq, sub_theme_freq, figures, output_path, gpt_model='gpt-3.5-turbo')

    print(f"\nAnalysis complete. Results saved in the '{output_path}' directory.\n")

    # Print summary
    print("Theme Frequency Summary:")
    print("-----------------------")
    for _, row in main_theme_freq.iterrows():
        print(f"{row['Theme']}: {row['Count']} entries")

if __name__ == "__main__":
    main()
