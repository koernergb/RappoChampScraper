import os
import json
import time
import asyncio
import datetime
from typing import List
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
from tiktoken import encoding_for_model
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import re
import cohere

# Load environment variables
load_dotenv()

class TokenChunker:
    def __init__(self, chunk_size: int = 3000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = encoding_for_model("gpt-4")
        
    def chunk(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = start + self.chunk_size - self.overlap
        return chunks

class Champion(BaseModel):
    name: str = Field(..., description="Full name of the engineering leader")
    role: str = Field(..., description="Job title/position")
    company: str = Field(..., description="Company name")
    url_found_at: str = Field(..., description="URL where this champion was found")

def clean_content(content: str) -> str:
    """Remove navigation, footer, and menu content"""
    lines = content.split('\n')
    
    skip_patterns = [
        r'^\s*\*\s*\[.*?\]\(.*?\)\s*$',  # Navigation links
        r'^\s*Toggle navigation\s*$',
        r'^\s*Menu\s*$',
        r'footer',
        r'navigation',
        r'Copyright',
        r'Terms',
        r'Privacy',
        r'Log[in|out]',
    ]
    
    cleaned_lines = [
        line for line in lines 
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns)
    ]
    
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content)
    
    return cleaned_content

class CaseStudyScraper:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.case_study_schema = {
            "name": "Case Study Link Extractor",
            "baseSelector": """
                a[href*='case-stud'],
                a[href*='customer'],
                a[href*='success-stor'],
                a[href*='testimonial'],
                a[href*='resource']
            """,
            "fields": [
                {"name": "url", "type": "attribute", "attribute": "href"},
                {"name": "title", "type": "text"}
            ]
        }

        self.content_schema = {
            "name": "Content Extractor",
            "baseSelector": """
                article,
                main,
                #main,
                #main-content,
                .main-content,
                .case-study,
                [class*='case-study'],
                [class*='casestudy'],
                .customer-story,
                [class*='customer-story'],
                [class*='success-story'],
                .content-wrapper,
                .content-container,
                .article-content,
                .post-content,
                [class*='content-'],
                [class*='-content'],
                .customer-content,
                .resource-content,
                [class*='customer-'],
                [class*='resource-'],
                [class*='story-'],
                [class*='-story'],
                [class*='testimonial'],
                section[class*='content'],
                section[class*='story'],
                section[class*='customer'],
                .container,
                .wrapper,
                body
            """,
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "headers", "type": "text", "selector": "h1, h2, h3, h4, h5, h6"},
                {"name": "quotes", "type": "text", "selector": "blockquote, .quote, [class*='quote']"},
                {"name": "testimonials", "type": "text", "selector": ".testimonial, [class*='testimonial']"},
                {"name": "html", "type": "html"},
                {"name": "structured_data", "type": "text", "selector": "script[type='application/ld+json']"}
            ],
            "exclude": {
                "selectors": [
                    "header", 
                    "footer", 
                    "nav",
                    ".nav",
                    ".navigation",
                    "#navigation",
                    ".menu",
                    "#menu",
                    ".sidebar",
                    "#sidebar",
                    ".footer",
                    "#footer",
                    ".header",
                    "#header",
                    ".cookie",
                    ".cookies",
                    ".modal",
                    ".popup",
                    ".ad",
                    ".advertisement",
                    "form",
                    ".form"
                ]
            }
        }

        self.llm_strategy = LLMExtractionStrategy(
            provider="command-r",
            api_token=os.getenv('COHERE_API_KEY'),
            schema=Champion.schema(),
            extraction_type="schema",
            instruction="""
            Goal: Extract engineering champions from this case study.
            
            A champion is someone who meets these criteria:
            1. Engineering role only (e.g., Software Engineering, Infrastructure, DevOps, Platform)
            2. Must be at one of these levels:
               - Staff/Principal Engineer
               - Engineering Manager or Senior Manager
               - Director/Senior Director of Engineering
               - VP of Engineering
               - CTO or similar technical C-level
            
            Return the data in this exact format:
            {
                "name": "Full Name",
                "role": "Exact Job Title",
                "company": "Company Name"
            }

            Do not nest the data under additional keys. If multiple champions are found, return an array of objects in the above format.
            """,
        )

        self.chunker = TokenChunker(
            chunk_size=3000,
            overlap=500
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def process_case_study(self, url: str, crawler: AsyncWebCrawler) -> List[Champion]:
        """Process a single case study URL with retries"""
        print(f"\nProcessing case study: {url}")
        champions = []
        
        try:
            print("Getting case study content...")
            content_result = await crawler.arun(
                url=url,
                extraction_strategy=JsonCssExtractionStrategy(self.content_schema),
                bypass_cache=True,
                wait_until="networkidle",
                timeout=30000
            )
            
            print(f"\nContent extraction results for {url}:")
            print(f"Success: {content_result.success}")
            # print(f"LOOK!: content_result: {content_result}")
            
            '''
            if content_result.response:
                print(f"Status: {content_result.response.status}")
                print(f"Headers: {dict(content_result.response.headers)}")

            if not content_result.success:
                print(f"Failed to get content for {url}")
                return []
            '''
            
            '''
            content_data = json.loads(content_result.extracted_content)
            print(f"\nExtracted data structure:")
            print(json.dumps(content_data, indent=2)[:1000] + "...")

            all_content = []
            
            main_content = " ".join(item.get('content', '') for item in content_data)
            if main_content:
                all_content.append(main_content)
                
            headers = " ".join(item.get('headers', '') for item in content_data)
            if headers:
                all_content.append(headers)
                
            quotes = " ".join(item.get('quotes', '') for item in content_data)
            testimonials = " ".join(item.get('testimonials', '') for item in content_data)
            if quotes:
                all_content.append(quotes)
            if testimonials:
                all_content.append(testimonials)

            page_content = " ".join(all_content)
            cleaned_content = clean_content(page_content)
            
            print(f"\nContent statistics:")
            print(f"Main content length: {len(main_content)}")
            print(f"Headers length: {len(headers)}")
            print(f"Quotes length: {len(quotes)}")
            print(f"Testimonials length: {len(testimonials)}")
            print(f"Total cleaned content length: {len(cleaned_content)}")
            '''
            
            '''
            if len(cleaned_content) < 100:
                print(f"Not enough content in {url}")
                html_content = next((item.get('html', '') for item in content_data), '')
                print(f"\nRaw HTML preview:")
                print(html_content[:1000] + "...")
                return []
            '''
            # Step 2: Minimal cleaning while preserving content
            raw_content = str(content_result)
            
            # Remove SVG definitions
            if "<defs>" in raw_content:
                raw_content = raw_content.split("<defs>")[0]
                
            # Remove scripts and styles
            if "<script" in raw_content:
                raw_content = raw_content.split("<script")[0]
                
            # Remove footers and navigation
            for section in ["footer", "nav", "header", "sidebar", "menu", "intercom"]:
                if section in raw_content.lower():
                    raw_content = raw_content.split(section)[0]

            # Remove common non-champion text
            for term in ["cookie", "privacy", "terms", "copyright", "subscribe", "newsletter"]:
                if term in raw_content.lower():
                    raw_content = raw_content.split(term)[0]
                
            # Basic HTML cleanup without regex
            raw_content = raw_content.replace("\\n", " ")
            raw_content = raw_content.replace("\\", "")
            raw_content = raw_content.replace('\"', '"')
            
            if len(raw_content) < 100:
                print(f"Not enough content in {url}")
                return []

            # Split into chunks using token-based chunking
            print("\nChunking content...")
            chunks = self.chunker.chunk(raw_content)
            
            all_champions = []
            print(f"Split content into {len(chunks)} chunks")

            def normalize_champion_data(data):
                if isinstance(data, dict):
                    if "Champion" in data:
                        data = data["Champion"]
                    if all(key in data for key in ["name", "role", "company"]):
                        return data
                return None

            for i, chunk in enumerate(chunks, 1):
                print(f"\nProcessing chunk {i}/{len(chunks)}")
                try:
                    champion_result = await crawler.arun(
                        url=url,
                        content=chunk,
                        extraction_strategy=self.llm_strategy,
                        bypass_cache=True,
                        wait_until="networkidle",
                        timeout=30000
                    )

                    if champion_result.success:
                        try:
                            extracted_data = json.loads(champion_result.extracted_content)
                            
                            if isinstance(extracted_data, list):
                                for item in extracted_data:
                                    normalized_data = normalize_champion_data(item)
                                    if normalized_data:
                                        normalized_data['url_found_at'] = url
                                        try:
                                            champion = Champion(**normalized_data)
                                            all_champions.append(champion)
                                            print(f"\nFound Champion in chunk {i}:")
                                            print(f"Name: {champion.name}")
                                            print(f"Role: {champion.role}")
                                            print(f"Company: {champion.company}")
                                        except ValidationError as ve:
                                            print(f"Validation error for data: {normalized_data}")
                                            print(f"Error details: {ve}")
                            elif isinstance(extracted_data, dict):
                                normalized_data = normalize_champion_data(extracted_data)
                                if normalized_data:
                                    normalized_data['url_found_at'] = url
                                    try:
                                        champion = Champion(**normalized_data)
                                        all_champions.append(champion)
                                        print(f"\nFound Champion in chunk {i}:")
                                        print(f"Name: {champion.name}")
                                        print(f"Role: {champion.role}")
                                        print(f"Company: {champion.company}")
                                    except ValidationError as ve:
                                        print(f"Validation error for data: {normalized_data}")
                                        print(f"Error details: {ve}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing champion data from chunk {i}: {e}")
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    if "rate_limit" in str(e).lower():
                        await asyncio.sleep(60)
                        continue

                await asyncio.sleep(2)

            seen = set()
            champions = []
            for champion in all_champions:
                key = (champion.name.lower(), champion.role.lower(), champion.company.lower())
                if key not in seen:
                    seen.add(key)
                    champions.append(champion)

            if champions:
                print(f"\nFound {len(champions)} unique champions in {url}:")
                for champion in champions:
                    print(f"Name: {champion.name}")
                    print(f"Role: {champion.role}")
                    print(f"Company: {champion.company}")
            else:
                print(f"No champions found in {url}")

            return champions
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
            if "rate_limit" in str(e).lower():
                raise
            
        return champions

    async def process_vendor(self, vendor: dict) -> List[Champion]:
        print(f"\nProcessing vendor: {vendor['name']}")
        all_champions = []
        failed_urls = []

        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                print(f"Getting case study URLs for {vendor['name']}...")
                links_result = await crawler.arun(
                    url=vendor['case_studies_url'],
                    extraction_strategy=JsonCssExtractionStrategy(self.case_study_schema),
                    bypass_cache=True,
                    wait_until="networkidle",
                    timeout=30000
                )

                if not links_result.success:
                    print(f"Failed to get case study links for {vendor['name']}")
                    return []

                links_data = json.loads(links_result.extracted_content)
                case_study_urls = set()
                
                for link in links_data:
                    if isinstance(link, dict) and 'url' in link:
                        url = link['url']
                        if not url.startswith('http'):
                            url = f"{vendor.get('base_url', vendor['case_studies_url'])}{url}"
                        if 'case-stud' in url or 'customer-testimonial' in url:
                            case_study_urls.add(url)

                print(f"Found {len(case_study_urls)} case studies for {vendor['name']}")

                checkpoint_file = f"data/checkpoint_{vendor['name'].lower()}.json"
                processed_urls = set()
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file) as f:
                        checkpoint = json.load(f)
                        processed_urls = set(checkpoint.get('processed_urls', []))
                        all_champions = [Champion(**c) for c in checkpoint.get('champions', [])]
                        print(f"Loaded {len(all_champions)} champions from checkpoint")

                remaining_urls = case_study_urls - processed_urls
                print(f"Processing {len(remaining_urls)} unprocessed case studies")

                for url in remaining_urls:
                    try:
                        champions = await self.process_case_study(url, crawler)
                        if champions:
                            all_champions.extend(champions)
                            processed_urls.add(url)
                            
                            checkpoint = {
                                'processed_urls': list(processed_urls),
                                'champions': [c.dict() for c in all_champions],
                                'failed_urls': failed_urls
                            }
                            
                            with open(checkpoint_file, 'w') as f:
                                json.dump(checkpoint, f, indent=2)
                            
                            print(f"Champions found so far in {vendor['name']}: {len(all_champions)}")
                            
                    except Exception as e:
                        print(f"Failed to process {url}: {e}")
                        failed_urls.append(url)
                        checkpoint = {
                            'processed_urls': list(processed_urls),
                            'champions': [c.dict() for c in all_champions],
                            'failed_urls': failed_urls
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint, f, indent=2)
                            
                    await asyncio.sleep(5)  # Rate limiting between case studies

        except Exception as e:
            print(f"Error processing vendor {vendor['name']}: {e}")

        print(f"Completed {vendor['name']}: Found {len(all_champions)} champions")
        if failed_urls:
            print(f"Failed to process {len(failed_urls)} URLs:")
            for url in failed_urls:
                print(f"- {url}")
            
        return all_champions

async def main():
    scraper = CaseStudyScraper("config.json")
    all_champions = []

    for vendor in scraper.config['vendors']:
        vendor_champions = await scraper.process_vendor(vendor)
        all_champions.extend(vendor_champions)
        
        # Save progress after each vendor
        os.makedirs('data', exist_ok=True)
        
        # Save timestamped JSON
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"data/champions_{timestamp}.json"
        pretty_json = json.dumps([c.dict() for c in all_champions], indent=2)
        
        with open(json_path, 'w') as f:
            f.write(pretty_json)
        
        print(f"\nCompleted {vendor['name']}")
        print("Champions found:")
        for champion in vendor_champions:
            print(f"- {champion.name} ({champion.role}) at {champion.company}")
            print(f"  Source: {champion.url_found_at}")
        
        await asyncio.sleep(10)  # Rate limiting between vendors

    # Final summary
    print(f"\nTotal champions found across all vendors: {len(all_champions)}")
    
    if all_champions:
        print("\nAll Champions found:")
        for champion in all_champions:
            print("\nChampion Details:")
            print(f"Name: {champion.name}")
            print(f"Role: {champion.role}")
            print(f"Company: {champion.company}")
            print(f"Found at: {champion.url_found_at}")
        
        print(f"\nFinal results saved to: {json_path}")
    else:
        print("No champions found!")

if __name__ == "__main__":
    asyncio.run(main())