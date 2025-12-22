/**
 * License Plate Recognition System - Main JavaScript
 */

// Global utilities
const LPR = {
    // Toast notification helper
    showToast(message, type = 'info', duration = 3000) {
        window.dispatchEvent(new CustomEvent('show-toast', {
            detail: { message, type, duration }
        }));
    },

    // Format date for display
    formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleDateString();
    },

    // Format time for display
    formatTime(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    // Format datetime
    formatDateTime(dateString) {
        if (!dateString) return '';
        return `${this.formatDate(dateString)} ${this.formatTime(dateString)}`;
    },

    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Copy text to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Copied to clipboard', 'success');
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            this.showToast('Failed to copy', 'error');
            return false;
        }
    }
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + U for upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            window.location.href = '/upload';
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            document.querySelectorAll('[x-data]').forEach(el => {
                if (el.__x) {
                    const data = el.__x.$data;
                    if (data.showAddModal) data.showAddModal = false;
                    if (data.selectedImage) data.selectedImage = null;
                }
            });
        }
    });

    console.log('LPR System initialized');
});

// Export for global use
window.LPR = LPR;
