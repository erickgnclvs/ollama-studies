// Main.js - Client-side functionality for Ollama Studies

document.addEventListener('DOMContentLoaded', function() {
    // Enable tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Enable markdown rendering in chat messages
    const markdownElements = document.querySelectorAll('.message-content');
    markdownElements.forEach(element => {
        // Look for code blocks and add syntax highlighting
        const codeBlocks = element.querySelectorAll('pre code');
        if (codeBlocks.length > 0 && typeof hljs !== 'undefined') {
            codeBlocks.forEach(block => {
                hljs.highlightBlock(block);
            });
        }
    });
    
    // Auto-resize text areas
    const autoResizeTextareas = document.querySelectorAll('textarea[data-autoresize]');
    autoResizeTextareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        // Trigger on page load
        textarea.dispatchEvent(new Event('input'));
    });
    
    // Add confirmation for destructive actions
    const confirmActions = document.querySelectorAll('[data-confirm]');
    confirmActions.forEach(element => {
        element.addEventListener('click', function(e) {
            const message = this.getAttribute('data-confirm') || 'Are you sure?';
            if (!confirm(message)) {
                e.preventDefault();
            }
        });
    });
});

// Function to copy code blocks to clipboard
function copyToClipboard(element) {
    const textToCopy = element.closest('.code-block').querySelector('code').textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
        // Show copied tooltip
        element.setAttribute('data-original-title', 'Copied!');
        const tooltip = bootstrap.Tooltip.getInstance(element);
        tooltip.show();
        
        // Reset tooltip after delay
        setTimeout(() => {
            element.setAttribute('data-original-title', 'Copy to clipboard');
        }, 1500);
    }).catch(err => {
        console.error('Could not copy text: ', err);
    });
}
