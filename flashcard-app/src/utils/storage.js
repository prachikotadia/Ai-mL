export const STORAGE_KEY = 'flashcard_app_progress';

export const getProgress = () => {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : {};
    } catch (e) {
        console.error("Failed to load progress", e);
        return {};
    }
};

export const saveProgress = (sectionId, data) => {
    try {
        const current = getProgress();
        const updated = {
            ...current,
            [sectionId]: {
                ...current[sectionId],
                ...data,
                lastUpdated: new Date().toISOString()
            }
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        return updated;
    } catch (e) {
        console.error("Failed to save progress", e);
    }
};

export const getSectionProgress = (sectionId) => {
    const all = getProgress();
    return all[sectionId] || { currentIndex: 0, knownCardIds: [] };
};
