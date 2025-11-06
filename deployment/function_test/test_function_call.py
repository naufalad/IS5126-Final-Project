#!/usr/bin/env python3
"""
Test script for /function_call API endpoint
"""

import requests
import json
import sys
import webbrowser
import time

BACKEND_URL = "http://127.0.0.1:8000"

def test_spotify_function():
    """Test Spotify link discovery"""
    print("=" * 60)
    print("🎵 Testing Spotify feature")
    print("=" * 60)
    
    email_data = {
        "subject": "Concert Announcement",
        "body": "Check out the new song 'Blinding Lights' by The Weeknd! Don't miss his upcoming concert in Singapore. Also check out 'Shape of You' by Ed Sheeran."
    }
    
    try:
        print(f"\n📧 Email:")
        print(f"  Subject: {email_data['subject']}")
        print(f"  Body: {email_data['body']}")
        print(f"\n🔄 Sending request to {BACKEND_URL}/function_call...")
        
        response = requests.post(
            f"{BACKEND_URL}/function_call",
            json=email_data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ Request succeeded!")
        print("\n📊 Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check Spotify results
        spotify_urls = []
        if result.get("success"):
            function_result = result.get("function_result", [])
            if isinstance(function_result, list):
                for func in function_result:
                    if func.get("function_name") == "spotify_link_discovery":
                        songs = func.get("data", {}).get("songs", [])
                        if songs:
                            print(f"\n🎵 Found {len(songs)} songs:")
                            for i, song in enumerate(songs, 1):
                                print(f"\n  {i}. {song.get('song')} by {song.get('artist')}")
                                spotify_url = song.get('spotify_url')
                                print(f"     🔗 Spotify: {spotify_url}")
                                if spotify_url:
                                    spotify_urls.append(spotify_url)
                                if song.get('album'):
                                    print(f"     💿 Album: {song.get('album')}")
                                if song.get('release_date'):
                                    print(f"     📅 Release date: {song.get('release_date')}")
                        else:
                            print("\n⚠️ No songs found")
            elif isinstance(function_result, dict):
                if function_result.get("function_name") == "spotify_link_discovery":
                    songs = function_result.get("data", {}).get("songs", [])
                    if songs:
                        print(f"\n🎵 Found {len(songs)} songs:")
                        for i, song in enumerate(songs, 1):
                            print(f"\n  {i}. {song.get('song')} by {song.get('artist')}")
                            spotify_url = song.get('spotify_url')
                            print(f"     🔗 Spotify: {spotify_url}")
                            if spotify_url:
                                spotify_urls.append(spotify_url)
        
        # Auto-open Spotify links
        if spotify_urls:
            print(f"\n🌐 Opening {len(spotify_urls)} Spotify links...")
            for i, url in enumerate(spotify_urls, 1):
                print(f"   Open {i}/{len(spotify_urls)}: {url}")
                webbrowser.open(url)
                if i < len(spotify_urls):
                    time.sleep(1)
            print("✅ All links opened in the browser!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to the backend server!")
        print("   Please ensure the backend service is running:")
        print("   cd deployment")
        print("   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout!")
        print("   LLM processing may take longer; please retry later")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calendar_function():
    """Test calendar event creation"""
    print("\n" + "=" * 60)
    print("📅 Testing calendar feature")
    print("=" * 60)
    
    email_data = {
        "subject": "Team Meeting",
        "body": "We have a team meeting scheduled for tomorrow at 2:00 PM in the main conference room. Please confirm your attendance."
    }
    
    try:
        print(f"\n📧 Email:")
        print(f"  Subject: {email_data['subject']}")
        print(f"  Body: {email_data['body']}")
        print(f"\n🔄 Sending request to {BACKEND_URL}/function_call...")
        
        response = requests.post(
            f"{BACKEND_URL}/function_call",
            json=email_data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ Request succeeded!")
        print("\n📊 Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("🚀 Function Call API Test Script")
    print("=" * 60)
    print("\n⚠️  Please ensure the backend service is running:")
    print("   cd deployment")
    print("   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    print("\n" + "-" * 60)
    
    # Check backend health
    try:
        health_check = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_check.status_code == 200:
            print("✅ Backend is healthy\n")
        else:
            print("⚠️  Backend health response abnormal\n")
    except:
        print("❌ Cannot connect to backend; please start the server first\n")
        sys.exit(1)
    
    # Run tests
    success = True
    
    # Test Spotify feature
    if not test_spotify_function():
        success = False
    
    # Test calendar feature (optional)
    # if not test_calendar_function():
    #     success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
