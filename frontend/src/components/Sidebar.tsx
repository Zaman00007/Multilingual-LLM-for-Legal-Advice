import React from 'react';
import { Conversation } from '../types';

interface SidebarProps {
  isOpen: boolean;
  conversations: Conversation[];
  currentConversation: Conversation | null;
  onSelectConversation: (conversation: Conversation) => void;
  onDeleteConversation: (conversationId: string) => void;
  onNewCase: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  conversations,
  currentConversation,
  onSelectConversation,
  onDeleteConversation,
  onNewCase,
}) => {
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => onSelectConversation(currentConversation!)}
        />
      )}
      
      {/* Sidebar */}
      <aside
        className={`absolute left-0 top-16 h-[calc(100vh-4rem)] w-80 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out z-30 lg:relative lg:translate-x-0 lg:top-0 lg:h-full ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-4 border-b border-gray-200">
            <button
              onClick={onNewCase}
              className="w-full flex items-center justify-center space-x-2 bg-legal-navy text-white px-4 py-3 rounded-lg hover:bg-blue-800 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span>New Case</span>
            </button>
          </div>
          
          {/* Conversations List */}
          <div className="flex-1 overflow-y-auto">
            {conversations.length === 0 ? (
              <div className="p-4 text-center text-legal-gray">
                <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p>No conversations yet</p>
                <p className="text-sm">Start a new case to begin</p>
              </div>
            ) : (
              <div className="p-2">
                {conversations.map((conversation) => (
                  <div
                    key={conversation.id}
                    className={`group relative p-3 rounded-lg cursor-pointer transition-colors ${
                      currentConversation?.id === conversation.id
                        ? 'bg-legal-navy text-white'
                        : 'hover:bg-gray-100'
                    }`}
                    onClick={() => onSelectConversation(conversation)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium truncate">{conversation.title}</h3>
                        <p className={`text-sm truncate ${
                          currentConversation?.id === conversation.id
                            ? 'text-blue-100'
                            : 'text-legal-gray'
                        }`}>
                          {conversation.messages.length} messages
                        </p>
                        <p className={`text-xs ${
                          currentConversation?.id === conversation.id
                            ? 'text-blue-200'
                            : 'text-gray-400'
                        }`}>
                          {formatDate(conversation.updatedAt)}
                        </p>
                      </div>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(conversation.id);
                        }}
                        className={`opacity-0 group-hover:opacity-100 p-1 rounded transition-opacity ${
                          currentConversation?.id === conversation.id
                            ? 'text-blue-200 hover:text-white'
                            : 'text-gray-400 hover:text-red-500'
                        }`}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Disclaimer */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="text-xs text-legal-gray leading-relaxed">
              <p className="font-medium mb-1">Disclaimer:</p>
              <p>
                AskLex provides general guidance only and is not a substitute for professional legal advice. 
                Always consult with a qualified attorney for your specific legal situation.
              </p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
