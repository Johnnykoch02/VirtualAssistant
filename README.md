# VIRTUAL ASSISTANT GUIDE

## OVERVIEW 

### Gwen the Virtual Assistant

Welcome to the repository for Gwen, a female virtual assistant agent who is the link between your voice activated prompts and your prefered compute platform. Through Python, Gwen is provided a platform to control and manage your desktop environment in hopes to perform tasks that you might otherwise find remidial.

## INSTALLATION GUIDE

We are currently working on building an installation for the project such that anyone can download and install Gwen with minimal experience with Python Programming.

### *Installation Guide coming Soon*

## DEVELOPMENT GUIDE

### Backend API

Curious on how it works? Gwen's job is to translate your commands into backend API calls in the form of JSON strings that mirror the following format:

```
{"target": "ClassName.FunctionTarget", ...}
```

She also includes a set of keyword arguments for the specific function.

Much of Gwen's backend implements AI technologies through the world-wide web, however we are currently developing many different useful AI models for speeding up the data pipeline. For more information regarding current model development, please reach out to:

    Jonathan Koch: johnnykoch02@gmail.com

### Backend Classes and Functions

Here's a look at the current featured backend classes and their respective function calls:

- Class Gwen:
  - clear_context(): Clears all running contexts and switches back to passive mode.
  - collect_keyword_data(int num_samples): Collects new user Keyword activations for the model to use and train.
  - output_speech(string text): Uses the Configured TTS Engine to speak the given text.

- Class YouTube:
  - play(string search_query, string channel_name = None): Searches and plays the top search result for the provided query and channel if present.

- Class Spotify:
  - play(string song, string artist): Plays a Song with a specified artist.
  - pause(): Pauses the music of the Spotify Bot.

- Class Netflix:
  - watch(string query): Plays a Show/Viewing for a specific target query.

### Usage Examples

Here's some examples of how Gwen translates some user commands into backend API calls:

1. "Open up Spotify and play Radioactive by Imagine Dragons":
    ```
    {"target": "Spotify.play", "song": "Radioactive", "artist": "Imagine Dragons"}
    ```
2. "Play 3Blue1Brown's video series about deep learning":
    ```
    {"target": "YouTube.play", "query": "Deep Learning", "channel_name": "3Blue1Brown"}
    ```
3. "Search YouTube for Funny Cat Videos":
    ```
    {"target": "YouTube.play", "query": "Funny Cat Videos"}
    ```
4. "Stop", "Turn this off", "Close the current application":
    ```
    {"target": "Gwen.clear_context"}
    ```
5. "What is Euler's Theorem?":
    ```
    {"target": "Gwen.output_speech", "text": "Euler's theorem, also known as Euler's identity, is a mathematical equation discovered by Swiss mathematician, physicist, and astronomer Leonhard Euler. It states that e^(i * pi) + 1 = 0, where e is the base of the natural logarithm (known as Euler's number), i is the imaginary unit, and pi is a ratio of a circle's circumference to its diameter."}
    ```

### Future Development

We have a list of features we want to add and improvements we're planning to make. We invite you to suggest more:

- LLM Server: Run a LLM for some provided weights path, receive GET REQUESTS and respond appropriately.
- LLM Trainer: Uses supervised learning to train input weights for some transformer.
- Gwen Repo: Expand backend capability, fix the Keyword model, potentially move the backend of this to Tinygrad, reduce API call delay, and more!
- Add a Gwen Backend: Prompt Processor for the user, can be trained using SL and/or RLHF via LoRA.

## Contribute

We welcome all contributions! If you have suggestions or want to report bugs, please don't hesitate to open an issue or submit a pull request.
