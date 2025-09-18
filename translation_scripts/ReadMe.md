# Translation Script Documentation

## Overview
This Python script was used to translate 852 Daoist texts from Chinese to English using OpenAI's GPT-4.1 API via OpenRouter. The translations were styled to match James Legge's classical Victorian prose approach.

## What the Script Does

### Core Functionality
- Reads Chinese texts from a JSON file (`docs.json`)
- Sends each text to GPT-4 with specific translation instructions
- Saves both original Chinese and English translations as Markdown files
- Processes multiple documents in parallel for efficiency
- Maintains progress tracking and can resume if interrupted

### Translation Approach
The script uses a detailed system prompt that:
1. Instructs GPT-4 to translate in James Legge's style (Victorian-era formal prose)
2. Provides extensive examples from Legge's actual Zhuangzi translations
3. Emphasizes preserving Markdown formatting (headers, lists, quotes)
4. Requires complete literal translation without summarization
5. Prohibits adding commentary or explanatory notes

### Technical Details
- **Parallel Processing**: Uses 20 concurrent workers to speed up translation
- **API**: OpenRouter API with GPT-4.1 model
- **Temperature**: Set to 0.72 for consistent but natural-sounding translations
- **Output Format**: Markdown files with preserved structure
- **Error Handling**: Automatically saves progress and can resume from interruptions

## Input File Structure

The `docs.json` file contains a list of dictionaries, each representing one document:

```json
[
  {
    "id": 1,
    "title": "弥罗宝诰",
    "content": "志心皈命礼。混元无极..."
  },
  {
    "id": 2,
    "title": "三清总诰",
    "content": "志心皈命礼。太极分三才..."
  },
  ...
]
```

Each document object contains:
- `id`: Unique document identifier (integer)
- `title`: Chinese title of the text
- `content`: Full Chinese text content

## Requirements
```python
openai
python-dotenv
```

## Configuration
The script requires an OpenRouter API key set in a `.env` file:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage
```bash
python translate.py
```