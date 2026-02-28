<template>
  <div class="chat-container">
    <!-- 顶部标题 -->
    <div class="header">
      <h1>医学知识问答系统</h1>
    </div>
    <!-- 聊天框 -->
    <div class="chat-box" ref="chatBox">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="[
          'message',
          message.role === 'user' ? 'user slide-in-right' : 'bot slide-in-left'
        ]"
      >
        <!-- 头像 -->
        <img :src="message.role === 'user' ? userAvatar : botAvatar" class="avatar" />
        <!-- 消息内容 -->
        <div class="message-content">
          <p v-html="message.text"></p>
        </div>
      </div>
    </div>
    <!-- 输入框和发送按钮 -->
    <div class="input-box">
      <input
        v-model="inputMessage"
        placeholder="请输入医学问题，例如‘感冒的症状是什么？’"
        @keyup.enter="sendQuery"
        :disabled="loading"
      />
      <button @click="sendQuery" :disabled="loading">
        {{ loading ? '发送中...' : '发送' }}
      </button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      inputMessage: '', // 用户输入的问题
      messages: [
        {
          role: 'bot',
          text: '<h2>欢迎使用医学知识问答系统！</h2><p>我是你的小助手，你可以问我以下问题：<br>- 感冒的症状是什么？<br>- 高血压的治疗方法有哪些？<br>- 糖尿病的病因是什么？<br>- 若是遇到我回答不出的问题，我会请教deepseek哦~</p>'
        }
      ], // 初始欢迎消息
      loading: false, // 加载状态
      userAvatar: '/user-avatar.png', // 用户头像路径
      botAvatar: '/bot-avatar.png' // 问答系统头像路径
    };
  },
  methods: {
    async sendQuery() {
      if (!this.inputMessage.trim()) return;

      // 添加用户消息
      this.messages.push({
        role: 'user',
        text: this.inputMessage
      });

      const query = this.inputMessage;
      this.inputMessage = ''; // 清空输入框
      this.loading = true; // 设置加载状态

      try {
        const response = await axios.post('http://localhost:5001/query', {
          query: query
        });
        console.log('后端响应:', response.data); // 调试日志
        if (response.data && response.data.answer) {
          this.messages.push({
            role: 'bot',
            text: response.data.answer
          });
        } else {
          throw new Error('后端响应中缺少 answer 字段');
        }
      } catch (error) {
        console.error('请求失败:', error); // 调试日志
        this.messages.push({
          role: 'bot',
          text: '无法回答您的问题，请尝试其他表述或咨询专业医生'
        });
      } finally {
        this.loading = false; // 无论成功或失败，都取消加载状态
        this.scrollToBottom(); // 滚动到底部
      }
    },
    scrollToBottom() {
      this.$nextTick(() => {
        const chatBox = this.$refs.chatBox;
        if (chatBox) {
          chatBox.scrollTop = chatBox.scrollHeight;
          console.log('已滚动到底部，scrollHeight:', chatBox.scrollHeight);
        } else {
          console.error('chatBox 未找到:', this.$refs.chatBox);
        }
      });
    }
  }
};
</script>

<style scoped>
.chat-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  padding: 20px;
  background-color: #f5f5f5;
}

.header {
  background-color: #007bff; /* 蓝色背景 */
  color: white; /* 白色字体 */
  padding: 10px;
  text-align: center;
  border-radius: 8px;
  margin-bottom: 20px;
}

.chat-box {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  margin-bottom: 20px;
}

.message {
  display: flex;
  margin-bottom: 15px;
  align-items: flex-start;
}

.message.user {
  flex-direction: row-reverse;
}

.message.bot {
  flex-direction: row;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  margin: 0 10px;
}

.message-content {
  max-width: 70%;
  padding: 10px;
  border-radius: 8px;
  background-color: #e6e6e6;
}

.message.user .message-content {
  background-color: #007bff;
  color: white;
}

.message.bot .message-content {
  background-color: #f0f0f0;
  color: #333;
}

.message-content p {
  margin: 0;
  word-wrap: break-word;
}

.input-box {
  display: flex;
  align-items: center;
  padding: 10px;
  background-color: #f0f0f0; /* 输入框底色 */
  border: 1px solid #ddd;
  border-radius: 8px;
}

.input-box input {
  flex: 1;
  padding: 10px;
  border: none;
  outline: none;
  font-size: 16px;
}

.input-box input:disabled {
  background-color: #e9ecef;
  cursor: not-allowed;
}

.input-box button {
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-box button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

/* 滑动动画 */
.slide-in-right {
  animation: slide-in-right 0.5s ease-in forwards;
}

.slide-in-left {
  animation: slide-in-left 0.5s ease-in forwards;
}

@keyframes slide-in-right {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slide-in-left {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
</style>