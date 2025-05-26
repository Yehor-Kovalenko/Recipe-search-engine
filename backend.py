from deep_translator import GoogleTranslator
from typing import List, Dict

import logging
import requests
from fastapi import HTTPException


def fetch_meals_by_ingredient(ingredient: str):
    url = f"https://www.themealdb.com/api/json/v1/1/filter.php?i={ingredient}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("meals", [])


def fetch_full_meal_details(meal_id: str):
    url = f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("meals", [])[0] if data.get("meals") else None


def meal_contains_all_ingredients(meal: dict, required_ingredients: List[str]) -> bool:
    meal_ingredients = set()

    # Extract up to 20 possible ingredients
    for i in range(1, 21):
        ingredient = meal.get(f"strIngredient{i}")
        if ingredient and ingredient.strip():
            meal_ingredients.add(ingredient.strip().lower())

    # Check if all required ingredients are present in the meal
    return all(ingredient.lower() in meal_ingredients for ingredient in required_ingredients)


def find_recipes_from_api(ingredients):
    if not ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")

        # Step 1: Get candidate meals using the first ingredient
    candidate_meals = fetch_meals_by_ingredient(ingredients[0])
    if not candidate_meals:
        return {"message": "No recipes found with the first ingredient", "recipes": []}

    # Step 2: Filter meals that contain all ingredients
    valid_recipes = []
    for meal in candidate_meals:
        full_meal = fetch_full_meal_details(meal["idMeal"])
        if not full_meal:
            continue

        if meal_contains_all_ingredients(full_meal, ingredients):
            valid_recipes.append({
                "idMeal": full_meal["idMeal"],
                "strMeal": full_meal["strMeal"],
                "strCategory": full_meal["strCategory"],
                "strArea": full_meal["strArea"],
                "strInstructions": full_meal["strInstructions"],
                "strMealThumb": full_meal["strMealThumb"],
                "ingredients": [
                    {
                        "ingredient": full_meal.get(f"strIngredient{i}"),
                        "measure": full_meal.get(f"strMeasure{i}")
                    }
                    for i in range(1, 21)
                    if full_meal.get(f"strIngredient{i}")
                ]
            })
    return valid_recipes


def translate_recipes_to_english(data: Dict, source_lang: str) -> Dict:
    """
    Translates a dictionary of recipes and their nested fields from the given source language to English.

    Args:
        data (Dict): The recipe data with structure:
                     {
                         recipes: [
                             {
                                 strMeal: str,
                                 strInstructions: str,
                                 ingredients: [
                                     { ingredient: str, measure: str }
                                 ]
                             }
                         ]
                     }
        source_lang (str): The original language code

    Returns:
        Dict: Translated dictionary with all text fields in English.
    """
    translator = GoogleTranslator(source='en', target=source_lang)

    translated_recipes = []
    for recipe in data.get("recipes", []):
        translated_meal = translator.translate(recipe.get("strMeal", ""))
        translated_instructions = translator.translate(recipe.get("strInstructions", ""))

        translated_ingredients = []
        for ing in recipe.get("ingredients", []):
            translated_ingredient = translator.translate(ing.get("ingredient", ""))
            translated_measure = translator.translate(ing.get("measure", ""))
            translated_ingredients.append({
                "ingredient": translated_ingredient,
                "measure": translated_measure
            })

        translated_recipes.append({
            "strMeal": translated_meal,
            "strInstructions": translated_instructions,
            "strMealThumb": recipe.get("strMealThumb", ""),
            "ingredients": translated_ingredients
        })

    return translated_recipes


def translate_to_english(ingredients, original_lang):
    translator = GoogleTranslator(source=original_lang, target='en')
    translated = []

    for item in ingredients:
        try:
            translated_text = translator.translate(item)
            translated.append(translated_text)
        except Exception as e:
            translated.append(f"(Error translating: {item})")
            print(f"Translation error for '{item}': {e}")

    return translated


def transcribe_audio_file(model, file_path):
    logging.info(f"Transcribing file: {file_path}")
    try:
        result = model.transcribe(file_path)
        logging.debug(f"Raw transcription result: {result}")
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        if not text:
            logging.warning("No speech detected in the audio.")
            return {
                "text": "(No speech detected)",
                "language": language
            }
        logging.info("Transcription complete.")
        return {
            "text": text,
            "language": language
        }
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise
