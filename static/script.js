// DOM Elements
const chatHistory = document.getElementById("chat-history");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const questionChips = document.querySelectorAll(".question-chip");
const tabButtons = document.querySelectorAll(".tab-btn");
const infoBtn = document.getElementById("info-btn");
const infoPopup = document.getElementById("info-popup");
const BACKEND_URL = ""; 

// Initialize with welcome message
window.addEventListener("DOMContentLoaded", () => {
    addMessage("AI", "Hello! I'm an AI assistant able access and explain Christoph's research. Ask me anything about his papers, methods, or whta ever you like to know.");
});

// Improved message function with typing indicator
function addMessage(sender, message, isTyping = false) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", sender);
    
    if (isTyping) {
        messageElement.innerHTML = `
            <strong>${sender}:</strong> 
            <span class="typing-indicator">
                <span>.</span><span>.</span><span>.</span>
            </span>
        `;
    } else {
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    }
    
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Show loading state
function showLoading() {
    addMessage("AI", "", true);
    return chatHistory.lastElementChild;
}

// Remove loading state
function removeLoading(loadingElement) {
    if (loadingElement) {
        loadingElement.remove();
    }
}

// Enhanced send function with loading state
async function sendQuestion() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage("You", question);
    userInput.value = "";
    
    const loadingElement = showLoading();
    
    try {
        const response = await fetch(`${BACKEND_URL}/process_input`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
            mode: 'cors'
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();
        removeLoading(loadingElement);
        addMessage("AI", data.response);
    } catch (error) {
        console.error("Error:", error);
        removeLoading(loadingElement);
        addMessage("AI", "I'm having trouble connecting to the research database. Please try again shortly.");
    }
}

// Quick question buttons
questionChips.forEach(chip => {
    chip.addEventListener("click", () => {
        const question = chip.getAttribute("data-question");
        userInput.value = question;
        sendQuestion();
    });
});

// Tab functionality
tabButtons.forEach(button => {
    button.addEventListener("click", () => {
        // Remove active class from all buttons
        tabButtons.forEach(btn => btn.classList.remove("active"));
        
        // Add active class to clicked button
        button.classList.add("active");
        
        // Hide all tab contents
        document.querySelectorAll(".tab-content").forEach(content => {
            content.classList.remove("active");
        });
        
        // Show selected tab content
        const tabId = button.getAttribute("data-tab");
        document.getElementById(tabId).classList.add("active");
    });
});

// Info popup toggle
infoBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    infoPopup.style.display = infoPopup.style.display === "block" ? "none" : "block";
});

document.querySelector(".close-btn").addEventListener("click", () => {
    infoPopup.style.display = "none";
});

// Close popup when clicking outside
document.addEventListener("click", (e) => {
    if (!infoPopup.contains(e.target) && e.target !== infoBtn) {
        infoPopup.style.display = "none";
    }
});

// Event listeners
sendButton.addEventListener("click", sendQuestion);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendQuestion();
});


document.addEventListener('DOMContentLoaded', function() {
    // Handle both publication and project cards
    const toggleCards = (containerClass, cardClass) => {
        const cards = document.querySelectorAll(`${containerClass} ${cardClass}`);
        
        cards.forEach(card => {
            const header = card.querySelector('.card-header, .project-header');
            
            header.addEventListener('click', () => {
                // Close all other open cards in this container
                document.querySelectorAll(`${containerClass} ${cardClass}.active`).forEach(otherCard => {
                    if (otherCard !== card) {
                        otherCard.classList.remove('active');
                    }
                });
                
                // Toggle current card
                card.classList.toggle('active');
            });
        });
    };

    toggleCards('.publication-grid', '.publication-card');
    toggleCards('.project-container', '.project-card');
});