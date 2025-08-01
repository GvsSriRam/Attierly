// Configuration
const API_BASE_URL = 'http://localhost:8000';
const USER_SERVICE_URL = 'http://localhost:8002';
const ECOMMERCE_SERVICE_URL = 'http://localhost:8003';

// Global variables
let userProfile = {
    gender: '',
    style: '',
    budget: ''
};

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusIndicator = document.getElementById('statusIndicator');
const aiStatus = document.getElementById('aiStatus');
const userStatus = document.getElementById('userStatus');
const ecommerceStatus = document.getElementById('ecommerceStatus');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadUserProfile();
    checkServices();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Auto-resize textarea (if we change to textarea later)
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
}

// Load user profile from localStorage
function loadUserProfile() {
    const savedProfile = localStorage.getItem('attierly_user_profile');
    if (savedProfile) {
        userProfile = JSON.parse(savedProfile);
        document.getElementById('gender').value = userProfile.gender;
        document.getElementById('style').value = userProfile.style;
        document.getElementById('budget').value = userProfile.budget;
    }
}

// Save user profile
function saveProfile() {
    userProfile = {
        gender: document.getElementById('gender').value,
        style: document.getElementById('style').value,
        budget: document.getElementById('budget').value
    };
    
    localStorage.setItem('attierly_user_profile', JSON.stringify(userProfile));
    
    // Show success message
    addMessage('Profile saved successfully! Your preferences will be used for future recommendations.', 'bot');
    
    // Update status
    updateStatus('Profile updated', 'success');
}

// Send message to chatbot
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    messageInput.value = '';
    
    // Show loading
    showLoading();
    updateStatus('Processing...', 'processing');
    
    try {
        // Get current user profile
        const currentProfile = {
            gender: document.getElementById('gender').value,
            style: document.getElementById('style').value,
            budget: document.getElementById('budget').value
        };
        
        // Create request payload with user profile
        const requestPayload = {
            user_message: message,
            user_id: 'frontend_user',
            session_id: 'web_session',
            task_type: 'recommendation',
            user_context: {
                gender_preference: currentProfile.gender || 'unknown',
                style_preference: currentProfile.style || 'casual',
                budget_range: currentProfile.budget || 'medium'
            }
        };
        
        console.log('Sending request with profile:', requestPayload);
        
        const response = await fetch(`${API_BASE_URL}/ai/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestPayload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add bot response to chat
        addMessage(data.response, 'bot');
        
        // Update status
        updateStatus('Ready', 'ready');
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
        updateStatus('Error', 'error');
    } finally {
        hideLoading();
    }
}

// Send quick message (for quick action buttons)
function sendQuickMessage(message) {
    messageInput.value = message;
    sendMessage();
}

// Add message to chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="${icon}"></i>
            <div class="message-text">
                ${formatMessage(text)}
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    
    // Smooth scroll to bottom
    setTimeout(() => {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// Format message text (handle line breaks, lists, etc.)
function formatMessage(text) {
    // Convert line breaks to <br> tags
    text = text.replace(/\n/g, '<br>');
    
    // Convert **text** to <strong>text</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *text* to <em>text</em>
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return text;
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Show loading overlay
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Update status indicator
function updateStatus(text, type) {
    const statusText = statusIndicator.querySelector('.status-text');
    const statusDot = statusIndicator.querySelector('.status-dot');
    
    statusText.textContent = text;
    
    // Remove existing classes
    statusDot.className = 'status-dot';
    
    // Add appropriate class based on type
    switch (type) {
        case 'ready':
            statusDot.style.background = '#4ade80';
            break;
        case 'processing':
            statusDot.style.background = '#f59e0b';
            break;
        case 'error':
            statusDot.style.background = '#ef4444';
            break;
        case 'success':
            statusDot.style.background = '#10b981';
            break;
        default:
            statusDot.style.background = '#4ade80';
    }
}

// Check services status
async function checkServices() {
    const services = [
        { name: 'AI Orchestrator', url: `${API_BASE_URL}/health`, element: aiStatus },
        { name: 'User Service', url: `${USER_SERVICE_URL}/health`, element: userStatus },
        { name: 'E-commerce Service', url: `${ECOMMERCE_SERVICE_URL}/health`, element: ecommerceStatus }
    ];
    
    for (const service of services) {
        try {
            const response = await fetch(service.url, { timeout: 5000 });
            if (response.ok) {
                service.element.classList.add('online');
            } else {
                service.element.classList.remove('online');
            }
        } catch (error) {
            console.error(`Error checking ${service.name}:`, error);
            service.element.classList.remove('online');
        }
    }
}

// Get AI configuration
async function getAIConfig() {
    try {
        const response = await fetch(`${API_BASE_URL}/ai/config`);
        if (response.ok) {
            const config = await response.json();
            console.log('AI Configuration:', config);
            return config;
        }
    } catch (error) {
        console.error('Error getting AI config:', error);
    }
}

// Utility function to show notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 1001;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Error handling utility
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    showNotification(`An error occurred: ${error.message}`, 'error');
}

// Add fetch timeout utility
function fetchWithTimeout(url, options = {}, timeout = 10000) {
    return Promise.race([
        fetch(url, options),
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Request timeout')), timeout)
        )
    ]);
}

// Export functions for global access
window.sendMessage = sendMessage;
window.sendQuickMessage = sendQuickMessage;
window.saveProfile = saveProfile;
window.checkServices = checkServices;
window.handleKeyPress = handleKeyPress; 