/**
 * License Plate Recognition System - API Client
 */

const API = {
    baseUrl: '/api',

    // Generic request method
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const mergedOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers,
            },
        };

        // Don't set Content-Type for FormData
        if (options.body instanceof FormData) {
            delete mergedOptions.headers['Content-Type'];
        }

        try {
            const response = await fetch(url, mergedOptions);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error?.message || 'Request failed');
            }

            return data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    },

    // GET request
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    },

    // POST request
    async post(endpoint, data) {
        const body = data instanceof FormData ? data : JSON.stringify(data);
        return this.request(endpoint, {
            method: 'POST',
            body,
        });
    },

    // PUT request
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },

    // DELETE request
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    },

    // ==========================================
    // Detection API
    // ==========================================

    async detectPlate(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        return this.post('/detect', formData);
    },

    async detectBatch(imageFiles) {
        const formData = new FormData();
        imageFiles.forEach(file => formData.append('images', file));
        return this.post('/detect/batch', formData);
    },

    // ==========================================
    // Detection History API
    // ==========================================

    async getDetections(params = {}) {
        return this.get('/detections', params);
    },

    async getDetection(id) {
        return this.get(`/detections/${id}`);
    },

    async deleteDetection(id) {
        return this.delete(`/detections/${id}`);
    },

    // ==========================================
    // Camera API
    // ==========================================

    async getCameras() {
        return this.get('/cameras');
    },

    async addCamera(camera) {
        return this.post('/cameras', camera);
    },

    async updateCamera(id, camera) {
        return this.put(`/cameras/${id}`, camera);
    },

    async deleteCamera(id) {
        return this.delete(`/cameras/${id}`);
    },

    // ==========================================
    // Statistics API
    // ==========================================

    async getStats() {
        return this.get('/stats');
    },

    // ==========================================
    // Settings API
    // ==========================================

    async getSettings() {
        return this.get('/settings');
    },

    async updateSettings(settings) {
        return this.put('/settings', settings);
    },

    // ==========================================
    // Health API
    // ==========================================

    async checkHealth() {
        return this.get('/health');
    },
};

// Export for global use
window.API = API;
