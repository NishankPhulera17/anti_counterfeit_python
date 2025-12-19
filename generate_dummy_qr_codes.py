#!/usr/bin/env python3
"""
Script to generate 1000 dummy QR codes with unique product IDs.
Calls the /generate_qr_cdp API endpoint for each product.
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"
GENERATE_ENDPOINT = f"{API_BASE_URL}/generate_qr_cdp"

# Generation settings
TOTAL_PRODUCTS = 1000
BATCH_SIZE = 10  # Number of concurrent requests
PRODUCT_ID_PREFIX = "PRODUCT"

def generate_product_id(index):
    """Generate a unique product ID."""
    return f"{PRODUCT_ID_PREFIX}{index:06d}"  # PRODUCT000001, PRODUCT000002, etc.

def generate_single_qr(product_id, index):
    """Generate a single QR code for a product ID."""
    try:
        payload = {
            "product_id": product_id
        }
        
        response = requests.post(
            GENERATE_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return {
                    "success": True,
                    "product_id": product_id,
                    "index": index,
                    "message": "Generated successfully"
                }
            else:
                return {
                    "success": False,
                    "product_id": product_id,
                    "index": index,
                    "message": result.get("message", "Unknown error")
                }
        else:
            return {
                "success": False,
                "product_id": product_id,
                "index": index,
                "message": f"HTTP {response.status_code}: {response.text[:100]}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "product_id": product_id,
            "index": index,
            "message": f"Request error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "product_id": product_id,
            "index": index,
            "message": f"Unexpected error: {str(e)}"
        }

def generate_qr_codes_batch(start_index, end_index):
    """Generate QR codes for a batch of product IDs."""
    results = []
    for i in range(start_index, end_index):
        product_id = generate_product_id(i)
        result = generate_single_qr(product_id, i)
        results.append(result)
        
        # Print progress for each item
        if result["success"]:
            print(f"✅ [{i+1}/{TOTAL_PRODUCTS}] Generated: {product_id}")
        else:
            print(f"❌ [{i+1}/{TOTAL_PRODUCTS}] Failed: {product_id} - {result['message']}")
    
    return results

def main():
    """Main function to generate 1000 QR codes."""
    print("="*70)
    print("QR Code Generation Script")
    print("="*70)
    print(f"Total products to generate: {TOTAL_PRODUCTS}")
    print(f"API endpoint: {GENERATE_ENDPOINT}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*70)
    print()
    
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print("✅ API server is reachable\n")
    except requests.exceptions.RequestException:
        print("⚠️  Warning: Could not reach API server. Make sure Flask app is running.")
        print(f"   Expected at: {API_BASE_URL}\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    start_time = time.time()
    all_results = []
    
    # Generate in batches using ThreadPoolExecutor for parallel processing
    print(f"Starting generation with {BATCH_SIZE} concurrent requests...\n")
    
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        # Submit all tasks
        futures = []
        for i in range(0, TOTAL_PRODUCTS, BATCH_SIZE):
            end_index = min(i + BATCH_SIZE, TOTAL_PRODUCTS)
            future = executor.submit(generate_qr_codes_batch, i, end_index)
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
    
    # Calculate statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    successful = sum(1 for r in all_results if r["success"])
    failed = TOTAL_PRODUCTS - successful
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total products: {TOTAL_PRODUCTS}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"📊 Average time per product: {elapsed_time/TOTAL_PRODUCTS:.3f} seconds")
    print(f"🚀 Generation rate: {TOTAL_PRODUCTS/elapsed_time:.2f} products/second")
    print("="*70)
    
    # Save results to file
    results_file = f"generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total": TOTAL_PRODUCTS,
                "successful": successful,
                "failed": failed,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # List failed products if any
    if failed > 0:
        print("\n⚠️  Failed Products:")
        for result in all_results:
            if not result["success"]:
                print(f"   - {result['product_id']}: {result['message']}")
    
    # List first and last successful products
    successful_products = [r for r in all_results if r["success"]]
    if successful_products:
        print(f"\n✅ First successful: {successful_products[0]['product_id']}")
        print(f"✅ Last successful: {successful_products[-1]['product_id']}")
        print(f"\n📁 Generated files are in:")
        print(f"   - generated_qr_cdp/ (combined QR+CDP images)")
        print(f"   - generated_cdp/ (reference CDP patterns)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

