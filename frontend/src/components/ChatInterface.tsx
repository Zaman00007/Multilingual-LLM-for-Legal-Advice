import React, { useState, useRef, useEffect } from "react";
import { Conversation, Message } from "../types";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";

interface ChatInterfaceProps {
  conversation: Conversation | null;
  onUpdateConversation: (messages: Message[]) => void;
  onNewCase: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  conversation,
  onUpdateConversation,
  onNewCase,
}) => {
  const [messages, setMessages] = useState<Message[]>(conversation?.messages || []);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (conversation?.messages) {
      setMessages(conversation.messages);
    }
  }, [conversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: "user",
      content: content.trim(),
      timestamp: new Date(),
    };

    const newMessages: Message[] = [...messages, userMessage];
    setMessages(newMessages);
    onUpdateConversation(newMessages);

    setIsLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: content }),
      });

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: "ai",
        content: data.answer,
        timestamp: new Date(),
      };

      const updatedMessages: Message[] = [...newMessages, botMessage];
      setMessages(updatedMessages);
      onUpdateConversation(updatedMessages);
    } catch (err) {
      const errorMsg: Message = {
        id: (Date.now() + 2).toString(),
        sender: "ai",
        content: "⚠️ Server error",
        timestamp: new Date(),
      };

      const errorMessages: Message[] = [...newMessages, errorMsg];
      setMessages(errorMessages);
      onUpdateConversation(errorMessages);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAdvice = (messageId: string) => {
    const message = messages.find((m) => m.id === messageId);
    if (message && message.sender === "ai") {
      console.log("Saving advice:", message.content);
      alert("Advice saved! (Check console for details)");
    }
  };

  if (!conversation) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center max-w-md">
          <div className="w-24 h-24 bg-legal-navy rounded-full flex items-center justify-center mx-auto mb-6">
            <svg
              className="w-12 h-12 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-legal-navy mb-2">
            Welcome to AskLex
          </h2>
          <p className="text-legal-gray mb-6">
            Describe your legal situation in natural language, and I'll provide
            guidance on possible next steps, potential risks, and general legal
            considerations.
          </p>
          <button
            onClick={onNewCase}
            className="bg-legal-navy text-white px-6 py-3 rounded-lg hover:bg-blue-800 transition-colors"
          >
            Start New Case
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-legal-gray mt-8">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
            <p className="text-lg font-medium">Start your conversation</p>
            <p className="text-sm">
              Describe your legal situation and I'll help guide you
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onSaveAdvice={handleSaveAdvice}
            />
          ))
        )}

        {isLoading && (
          <div className="flex justify-start">
            <div className="ai-bubble chat-bubble p-4">
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="border-t border-gray-200 p-6 bg-white">
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default ChatInterface;
