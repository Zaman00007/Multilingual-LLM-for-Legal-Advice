export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  isSaved?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ApiResponse {
  message: string;
  success: boolean;
  error?: string;
}

export interface SavedAdvice {
  id: string;
  messageId: string;
  content: string;
  savedAt: Date;
}
