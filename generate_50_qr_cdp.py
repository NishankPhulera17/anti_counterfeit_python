#!/usr/bin/env python3
"""
Script to generate 50 QR+CDP codes for PRODUCT1 through PRODUCT50
"""
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8000/generate_qr_cdp"

def generate_qr_cdp(product_id):
    """Generate QR+CDP for a given product_id"""
    try:
        response = requests.post(
            API_URL,
            json={"product_id": product_id},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success":
            return {
                "success": True,
                "product_id": product_id,
                "serial_id": result.get("serial_id"),
                "cdp_id": result.get("cdp_id"),
                "message": result.get("message")
            }
        else:
            return {
                "success": False,
                "product_id": product_id,
                "error": result.get("message", "Unknown error")
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "product_id": product_id,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "product_id": product_id,
            "error": f"Unexpected error: {str(e)}"
        }

def main():
    print(f"[INFO] Starting generation of 50 QR+CDP codes at {datetime.now()}")
    print("=" * 60)
    
    results = []
    success_count = 0
    failure_count = 0
    
    # Generate for PRODUCT1 through PRODUCT50
    for i in range(1, 51):
        product_id = f"PRODUCT{i}"
        print(f"[{i}/50] Generating QR+CDP for {product_id}...", end=" ", flush=True)
        
        result = generate_qr_cdp(product_id)
        results.append(result)
        
        if result["success"]:
            success_count += 1
            print(f"✓ Success - Serial ID: {result['serial_id']}")
        else:
            failure_count += 1
            print(f"✗ Failed - {result['error']}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Generation complete!")
    print(f"  Total: 50")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    print("=" * 60)
    
    # Save results to file
    output_file = "generation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_file}")
    
    # Print failed products if any
    if failure_count > 0:
        print("\n[FAILED PRODUCTS]")
        for result in results:
            if not result["success"]:
                print(f"  - {result['product_id']}: {result['error']}")

if __name__ == "__main__":
    main()

