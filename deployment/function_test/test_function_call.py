#!/usr/bin/env python3
"""
测试 /function_call API 端点的脚本
"""

import requests
import json
import sys
import webbrowser
import time

BACKEND_URL = "http://127.0.0.1:8000"

def test_spotify_function():
    """测试 Spotify 链接发现功能"""
    print("=" * 60)
    print("🎵 测试 Spotify 功能")
    print("=" * 60)
    
    email_data = {
        "subject": "Concert Announcement",
        "body": "Check out the new song 'Blinding Lights' by The Weeknd! Don't miss his upcoming concert in Singapore. Also check out 'Shape of You' by Ed Sheeran."
    }
    
    try:
        print(f"\n📧 邮件内容:")
        print(f"  主题: {email_data['subject']}")
        print(f"  正文: {email_data['body']}")
        print(f"\n🔄 发送请求到 {BACKEND_URL}/function_call...")
        
        response = requests.post(
            f"{BACKEND_URL}/function_call",
            json=email_data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ 请求成功!")
        print("\n📊 响应结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 检查是否有 Spotify 结果
        spotify_urls = []
        if result.get("success"):
            function_result = result.get("function_result", [])
            if isinstance(function_result, list):
                for func in function_result:
                    if func.get("function_name") == "spotify_link_discovery":
                        songs = func.get("data", {}).get("songs", [])
                        if songs:
                            print(f"\n🎵 找到 {len(songs)} 首歌曲:")
                            for i, song in enumerate(songs, 1):
                                print(f"\n  {i}. {song.get('song')} by {song.get('artist')}")
                                spotify_url = song.get('spotify_url')
                                print(f"     🔗 Spotify: {spotify_url}")
                                if spotify_url:
                                    spotify_urls.append(spotify_url)
                                if song.get('album'):
                                    print(f"     💿 专辑: {song.get('album')}")
                                if song.get('release_date'):
                                    print(f"     📅 发布日期: {song.get('release_date')}")
                        else:
                            print("\n⚠️ 未找到歌曲")
            elif isinstance(function_result, dict):
                if function_result.get("function_name") == "spotify_link_discovery":
                    songs = function_result.get("data", {}).get("songs", [])
                    if songs:
                        print(f"\n🎵 找到 {len(songs)} 首歌曲:")
                        for i, song in enumerate(songs, 1):
                            print(f"\n  {i}. {song.get('song')} by {song.get('artist')}")
                            spotify_url = song.get('spotify_url')
                            print(f"     🔗 Spotify: {spotify_url}")
                            if spotify_url:
                                spotify_urls.append(spotify_url)
        
        # 自动打开 Spotify 链接
        if spotify_urls:
            print(f"\n🌐 正在自动打开 {len(spotify_urls)} 个 Spotify 链接...")
            for i, url in enumerate(spotify_urls, 1):
                print(f"   打开链接 {i}/{len(spotify_urls)}: {url}")
                webbrowser.open(url)
                if i < len(spotify_urls):  # 避免最后一个链接后也等待
                    time.sleep(1)  # 延迟1秒，避免浏览器打开太快
            print("✅ 所有链接已在浏览器中打开!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到后端服务器!")
        print("   请确保后端服务正在运行:")
        print("   cd deployment")
        print("   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时!")
        print("   LLM 处理可能需要更长时间，请稍后重试")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP 错误: {e.response.status_code}")
        print(f"   响应: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calendar_function():
    """测试日历事件创建功能"""
    print("\n" + "=" * 60)
    print("📅 测试日历功能")
    print("=" * 60)
    
    email_data = {
        "subject": "Team Meeting",
        "body": "We have a team meeting scheduled for tomorrow at 2:00 PM in the main conference room. Please confirm your attendance."
    }
    
    try:
        print(f"\n📧 邮件内容:")
        print(f"  主题: {email_data['subject']}")
        print(f"  正文: {email_data['body']}")
        print(f"\n🔄 发送请求到 {BACKEND_URL}/function_call...")
        
        response = requests.post(
            f"{BACKEND_URL}/function_call",
            json=email_data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ 请求成功!")
        print("\n📊 响应结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("🚀 Function Call API 测试脚本")
    print("=" * 60)
    print("\n⚠️  请确保后端服务正在运行:")
    print("   cd deployment")
    print("   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    print("\n" + "-" * 60)
    
    # 检查后端是否运行
    try:
        health_check = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_check.status_code == 200:
            print("✅ 后端服务运行正常\n")
        else:
            print("⚠️  后端服务响应异常\n")
    except:
        print("❌ 无法连接到后端服务，请先启动后端\n")
        sys.exit(1)
    
    # 运行测试
    success = True
    
    # 测试 Spotify 功能
    if not test_spotify_function():
        success = False
    
    # 测试日历功能（可选）
    # if not test_calendar_function():
    #     success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试完成!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
