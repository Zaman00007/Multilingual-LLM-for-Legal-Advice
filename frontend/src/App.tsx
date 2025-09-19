import React, { useState } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { Conversation, Message } from './types';

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const startNewCase = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: `New Case ${conversations.length + 1}`,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    
    setConversations(prev => [newConversation, ...prev]);
    setCurrentConversation(newConversation);
    setIsSidebarOpen(false);
  };

  const selectConversation = (conversation: Conversation) => {
    setCurrentConversation(conversation);
    setIsSidebarOpen(false);
  };

  const updateConversation = (messages: Message[]) => {
    if (!currentConversation) return;
    
    const updatedConversation = {
      ...currentConversation,
      messages,
      updatedAt: new Date(),
    };
    
    setCurrentConversation(updatedConversation);
    setConversations(prev => 
      prev.map(conv => 
        conv.id === currentConversation.id ? updatedConversation : conv
      )
    );
  };

  const deleteConversation = (conversationId: string) => {
    setConversations(prev => prev.filter(conv => conv.id !== conversationId));
    if (currentConversation?.id === conversationId) {
      setCurrentConversation(null);
    }
  };

  return (
    <div className="h-screen bg-legal-light overflow-hidden">
      <Header 
        onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewCase={startNewCase}
      />
      
      <div className="flex h-full pt-16">
        <Sidebar
          isOpen={isSidebarOpen}
          conversations={conversations}
          currentConversation={currentConversation}
          onSelectConversation={selectConversation}
          onDeleteConversation={deleteConversation}
          onNewCase={startNewCase}
        />
        
        <main className={`flex-1 flex flex-col transition-all duration-300 ${
          isSidebarOpen ? 'lg:ml-80' : 'lg:ml-0'
        }`}>
          <ChatInterface
            conversation={currentConversation}
            onUpdateConversation={updateConversation}
            onNewCase={startNewCase}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
