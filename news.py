import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BartTokenizer

# Function to fetch news articles from NewsAPI
def fetch_newsapi_articles(api_key, keywords, num_articles):
    url = 'https://newsapi.org/v2/everything'  # API endpoint for fetching news articles
    params = {
        'q': ' OR '.join(keywords),  # Query parameters for searching keywords
        'apiKey': api_key,  # API key for authentication
        'pageSize': num_articles  # Number of articles to retrieve
    }
    response = requests.get(url, params=params)
    return response.json().get('articles', [])  # Return the list of articles or an empty list

# Function to scrape the full content of an article
def scrape_full_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [para.text for para in soup.find_all('p')]  # Extract all paragraph text
    return ' '.join(paragraphs)  # Combine paragraphs into a single text block

# Function to chunk the article text into smaller parts for summarization
def chunk_text(text, tokenizer, max_length):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)  # Tokenize the text
    input_ids = inputs["input_ids"][0]  # Extract token IDs
    # Split token IDs into chunks of max_length
    return [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]

# Function to summarize the article content
def summarize_article(article_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)  # Load the tokenizer
    summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer)  # Load the summarization pipeline

    max_token_length = 512  # Maximum length for each chunk
    token_chunks = chunk_text(article_text, tokenizer, max_token_length)  # Split the article into chunks

    summarized_chunks = []
    for chunk in token_chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        
        # Calculate dynamic max_length and min_length based on the input length
        input_length = len(decoded_chunk.split())
        max_length = min(150, input_length)  # Ensure max_length doesn’t exceed the input length
        min_length = min(max(50, input_length // 2), max_length - 10)  # Set min_length appropriately

        # Generate the summary for the current chunk
        summary = summarizer(decoded_chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summarized_chunks.append(summary)
    
    final_summary = ' '.join(summarized_chunks)  # Combine all summarized chunks into the final summary
    return final_summary

def main():
    print("Welcome to the News AI Summarizer")
    print("This program will fetch and summarize the latest news articles based on your keywords.")

    while True:
        # Get user input for keywords and number of articles
        keywords = input("Please enter the keywords for the news you're interested in (comma-separated, e.g., technology, AI): ").split(',')
        keywords = [keyword.strip() for keyword in keywords]  # Clean up whitespace
        num_articles = int(input("How many articles would you like to retrieve? "))

        api_key = '73c548975c2342f9b69c8c533b73b6dc'  # Replace with your actual NewsAPI key
        articles = fetch_newsapi_articles(api_key, keywords, num_articles)

        for article in articles:
            title = article.get('title', 'No Title')
            description = article.get('description', '')
            url = article.get('url', '')
            content = scrape_full_content(url)
            
            print(f"\nTitle: {title}")
            print(f"Description: {description}")
            print(f"Content Preview: {content[:100]}...")  # Show a preview of the content

            if content:
                summary = summarize_article(content)  # Summarize the article content
                print(f"Summary: {summary}")
            
            print("---------------")

        # Ask if the user wants to search for more articles
        choice = input("\nWould you like to search for more articles? (yes/no): ").lower()
        if choice != 'yes':
            print("Thank you for using the News AI Summarizer. Goodbye!")
            break

if __name__ == "__main__":
    main()
