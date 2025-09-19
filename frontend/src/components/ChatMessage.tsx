import React from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
  onSaveAdvice: (messageId: string) => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onSaveAdvice }) => {
  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const formatContent = (content?: string) => {
    if (!content) return null; // ✅ safeguard

    return content
      .split('\n')
      .map((line, index) => {
        if (line.startsWith('**') && line.endsWith('**')) {
          return (
            <strong key={index} className="font-semibold">
              {line.slice(2, -2)}
            </strong>
          );
        }
        return line;
      })
      .map((line, index) => (
        <div key={index} className="mb-2 last:mb-0">
          {line}
        </div>
      ));
  };

  if (message.sender === 'user') {
    return (
      <div className="flex justify-end">
        <div className="user-bubble chat-bubble p-4 shadow-sm">
          <div className="text-sm">{message.content || ''}</div> {/* ✅ safeguard */}
          <div className="text-xs text-blue-100 mt-2 opacity-70">
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="ai-bubble chat-bubble p-4 shadow-sm relative group">
        <div className="text-sm leading-relaxed">
          {formatContent(message.content)}
        </div>

        <div className="flex items-center justify-between mt-3">
          <div className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </div>

          <button
            onClick={() => onSaveAdvice(message.id)}
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-gray-100"
            title="Save this advice"
          >
            <svg
              className="w-4 h-4 text-legal-gray"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
