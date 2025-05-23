# Integrating the Updated Word Similarity API into Your Lovable Game

This guide explains how to adapt your Lovable game to use the enhanced features of the word similarity API. The API endpoint `/similarity?word1=...&word2=...` now provides additional information about how close a guessed word is to a target word.

## 1. Understanding the New API Response

When you make a GET request to the `/similarity` endpoint (e.g., `/similarity?word1=player_guess&word2=secret_word`), the JSON response has been updated. Besides the existing `similarity` score, it now includes:

*   `is_in_top_1000` (boolean):
    *   This field will be `true` if `word1` (the guessed word) is among the 1000 words most similar to `word2` (the secret/target word).
    *   It will be `false` otherwise.
*   `rank_in_top_1000` (integer or `null`):
    *   If `is_in_top_1000` is `true`, this field contains the rank of `word1` relative to `word2`. The rank will be an integer from 1 to 1000 (e.g., 1 means it's the most similar word among the top 1000, and 1000 means it's the 1000th most similar).
    *   If `is_in_top_1000` is `false`, this field will be `null`.

**Example JSON Response:**

```json
{
  "word1": "dog",
  "word2": "canine",
  "similarity": 0.85,
  "is_in_top_1000": true,
  "rank_in_top_1000": 5
}
```

Or, if the word is not in the top 1000:

```json
{
  "word1": "banana",
  "word2": "canine",
  "similarity": 0.12,
  "is_in_top_1000": false,
  "rank_in_top_1000": null
}
```

## 2. Prompt/Instructions for Lovable

Here's a prompt you can use to instruct Lovable (or a similar AI game development platform) to integrate these changes:

---

"Hey Lovable,

I've updated my word similarity API. The GET request to `/similarity` with `word1` (the player's guessed word) and `word2` (the secret target word) now returns a JSON response with two new fields:

*   `is_in_top_1000`: A boolean value. It's `true` if `word1` is in the top 1000 most similar words to `word2`, and `false` otherwise.
*   `rank_in_top_1000`: An integer representing the rank (from 1 to 1000) if `is_in_top_1000` is `true`. It will be `null` if `is_in_top_1000` is `false`.

Please update the game's mechanics as follows:

1.  **Parse New Fields:** When the game receives the response from the `/similarity` API call, ensure it correctly parses `is_in_top_1000` and `rank_in_top_1000` in addition to the existing `similarity` score.

2.  **Conditional Player Feedback:**
    *   After a player makes a guess, check the `is_in_top_1000` field from the API response.
    *   If `is_in_top_1000` is `true`:
        *   Display a message to the player, for example: "Hot! Your guess is in the top 1000 closest words!"
        *   Also, display the rank: "Rank: [rank_in_top_1000]".
        *   Consider changing the color of the guessed word in the list of previous guesses to green, or adding a special icon next to it.
    *   If `is_in_top_1000` is `false`:
        *   You might display a message like: "Getting colder... that word is not in the top 1000." or simply no rank-related message.
        *   Ensure the regular similarity score is still displayed as before.

3.  **Maintain Existing Functionality:** The game should continue to display the main similarity score (e.g., "Similarity: [similarity_value]") regardless of whether the word is in the top 1000 or not. The game should also continue to handle cases where words are not found in the vocabulary.

Could you implement these changes to enhance the player's feedback?"

---

## 3. Example of Game Logic Change (Conceptual)

Here's a simplified, conceptual way your game logic (which Lovable would help implement) might adapt:

**Previous Logic (Simplified):**

```
function handlePlayerGuess(guessedWord, secretWord):
  apiResponse = callSimilarityApi(guessedWord, secretWord)
  similarityScore = apiResponse.similarity
  
  display "Your guess: " + guessedWord
  display "Similarity: " + similarityScore
  
  if similarityScore is very high:
    display "You win!"
```

**New Logic with Rank Information (Simplified):**

```
function handlePlayerGuess(guessedWord, secretWord):
  apiResponse = callSimilarityApi(guessedWord, secretWord)
  
  // Existing data
  similarityScore = apiResponse.similarity 
  
  // New data from API
  isInTop1000 = apiResponse.is_in_top_1000
  rankInTop1000 = apiResponse.rank_in_top_1000
  
  display "Your guess: " + guessedWord
  display "Similarity: " + similarityScore // Still show the direct similarity
  
  if isInTop1000 is true:
    display "Great guess! It's in the top 1000!"
    display "Rank: " + rankInTop1000
    // Optional: Change style of 'guessedWord' in UI (e.g., make text green)
  else:
    display "That word is not among the top 1000 closest."
    // Optional: Change style of 'guessedWord' in UI (e.g., make text orange/red)

  if similarityScore is very high: // Or perhaps if rankInTop1000 == 1
    display "You win!" 
```

This updated feedback mechanism will provide players with more nuanced information about their guesses, making the game more engaging. Remember to test thoroughly after Lovable implements these changes!
