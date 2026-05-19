# data_loader.py
import ast
import pandas as pd
from llama_index.core import Document


def load_documents(csv_path: str = "indonesian_recipes.csv") -> list[Document]:
    df = pd.read_csv(csv_path)
    df["title"] = df["title"].str.replace(r'[^\w\s]', '', regex=True)
    df = df.drop_duplicates(subset=["title"], keep="first")
    documents = []

    for _, row in df.iterrows():
        ingredients = row["ingredients"]
        steps = row["steps"]

        # Konversi string representation of list -> Python list
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except Exception:
                ingredients = [ingredients]

        if isinstance(steps, str):
            try:
                steps = ast.literal_eval(steps)
            except Exception:
                steps = [steps]

        ingredients_text = "\n".join(
            [f"- {item}" for item in ingredients]
        )

        steps_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(steps)]
        )

        text = f"""
Recipe Title: {row['title']}

Ingredients:
{ingredients_text}

Cooking Steps:
{steps_text}
""".strip()

        metadata = {
            "title": row["title"],
            "num_ingredients": int(row["num_ingredients"]),
            "num_steps": int(row["num_steps"]),
            "char_count": int(row["char_count"]),
        }

        documents.append(
            Document(
                text=text,
                metadata=metadata,
            )
        )

    return documents