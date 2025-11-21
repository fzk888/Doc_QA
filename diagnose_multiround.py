#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多轮对话功能诊断脚本
验证所有改动是否正确应用
"""

import os
import re

def check_file_changes():
    """检查文件改动"""
    base_path = r'd:\大模型应用开发\RAG\Doc_QA'
    
    checks = [
        # functions.py 检查
        {
            'file': 'functions.py',
            'checks': [
                ('history_str 构建代码', 'history_items.append'),
                ('multiple_dialogue 检查', 'if multiple_dialogue and len(input_query) > 1:'),
                ('getattr role 检查', 'role = getattr(msg, \'role\', \'user\')'),
                ('统一日志器配置', 'logger.propagate = False')
            ]
        },
        # app.py 检查
        {
            'file': 'app.py',
            'checks': [
                ('only_llm 参数修复', 'only_llm(query, prompt_template_from_user, temperature, multiple_dialogue)'),
                ('请求ID中间件', '[req:'),
                ('mulitdoc_qa 接收日志', '/mulitdoc_qa received kb='),
                ('mulitdoc_qa 标志日志', 'flags only_chatKBQA=')
            ]
        },
        # index.html 检查
        {
            'file': 'server/index.html',
            'checks': [
                ('conversationHistory 变量', 'let conversationHistory = []'),
                ('addMessage 参数', 'function addMessage(role, content, isPlaceholder = false)'),
                ('isPlaceholder 检查', 'if (!isPlaceholder)'),
                ('multiple_dialogue 标志', 'multiple_dialogue: conversationHistory.length > 1'),
            ]
        },
        # documen_processing.py 检查
        {
            'file': 'documen_processing.py',
            'checks': [
                ('DOCX Markdown 图片识别', 'img_md_pattern'),
                ('DOCX ZIP 媒体回退', 'zipfile.ZipFile(doc_file)'),
                ('OCR 请求调用', 'requests.post(url_f')
            ]
        }
    ]
    
    print("=" * 60)
    print("多轮对话功能诊断")
    print("=" * 60)
    
    all_passed = True
    
    for file_check in checks:
        file_path = os.path.join(base_path, file_check['file'])
        print(f"\n📄 检查文件: {file_check['file']}")
        
        if not os.path.exists(file_path):
            print(f"  ❌ 文件不存在: {file_path}")
            all_passed = False
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for check_name, check_pattern in file_check['checks']:
                if check_pattern in content:
                    print(f"  ✅ {check_name}")
                else:
                    print(f"  ❌ {check_name}")
                    print(f"     期望找到: {check_pattern}")
                    all_passed = False
        
        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✨ 所有检查通过！多轮对话功能已正确实现")
    else:
        print("⚠️  某些检查未通过，请检查改动")
    print("=" * 60)
    
    return all_passed

def suggest_next_steps():
    """建议后续步骤"""
    print("\n🚀 后续步骤:")
    print("""
1. 启动后端服务:
   python app.py

2. 打开前端界面:
   在浏览器访问 http://localhost:8000

3. 测试多轮对话:
   - 发送第一个问题: "什么是机器学习？"
   - 发送追问: "那么深度学习有什么不同？"
   - 观察LLM是否能引用之前的内容

4. 使用API测试页面 (可选):
   打开 test_multiround_api.html

5. 查看详细改动说明:
   打开 MULTIROUND_CHANGELOG.md
    """)

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # Windows支持
    
    passed = check_file_changes()
    suggest_next_steps()
    
    sys.exit(0 if passed else 1)
