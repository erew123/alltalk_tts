import asyncio
import aiohttp
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

async def send_tts_request(session, url, text, request_id):
    start_time = time.time()
    try:
        async with session.post(url, data={
            "text_input": text,
            "text_filtering": "standard",
            "character_voice_gen": "en_US-ljspeech-high.onnx",
            "narrator_enabled": "false",
            "narrator_voice_gen": "en_US-ljspeech-high.onnx",
            "text_not_inside": "character",
            "language": "en",
            "output_file_name": f"loadtest_{request_id}",
            "output_file_timestamp": "true",
            "autoplay": "false",
            "autoplay_volume": "0.8"
        }) as response:
            result = await response.json()
            end_time = time.time()
            print(f"Request {request_id} completed in {end_time - start_time:.2f} seconds.")
            print(f"Response: {result}")
            return result
    except Exception as e:
        print(f"Request {request_id} failed: {str(e)}")
        return None

async def run_load_test(num_requests, text_length, api_url):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            text = f"This is a test of the MEM load balancing system. Request number {i+1}. " * (text_length // 10)
            task = asyncio.ensure_future(send_tts_request(session, api_url, text, i+1))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

def main():
    parser = argparse.ArgumentParser(description="Load test for MEM TTS API")
    parser.add_argument("--requests", type=int, default=5, help="Number of simultaneous requests to send")
    parser.add_argument("--length", type=int, default=50, help="Approximate length of text for each request")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:7401/api/tts-generate", help="URL of the MEM API")
    args = parser.parse_args()

    print(f"AllTalk MEM load tester.")
    print(f"\nYou can use this to load test MEM. You can specify:")
    print(f"\n  - the amount of simultaneous request to send --requests")
    print(f"  - the amount of character text per request   --length")
    print(f"  - the URL to send the request to             --url")
    print(f"\nA full command line would look like this")
    print(f"\npython mem_load_test.py --requests 3 --length 50 --url `http://127.0.0.1:7401/api/tts-generate`")
    print(f"\nAll requests are sent simultaniously at the same time. As such the 'completed in' time is")
    print(f"calculated from the time the requests were sent, not the actual TTS processing time per")
    print(f"individual request within the TTS engine.")
    print(f"\nStarting load test with {args.requests} simultaneous requests...")
    print(f"Each request will have approximately {args.length} characters of text.")
    print(f"Sending requests to: {args.url}\n")

    start_time = time.time()
    
    # Run the async function in a separate thread to avoid blocking
    with ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, run_load_test(args.requests, args.length, args.url))
        results = future.result()

    end_time = time.time()

    successful_requests = sum(1 for result in results if result and result.get("status") == "generate-success")
    failed_requests = sum(1 for result in results if result is None or result.get("status") != "generate-success")

    print(f"\nLoad test completed in {end_time - start_time:.2f} seconds.")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")

if __name__ == "__main__":
    main()