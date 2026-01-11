/**
 * Sidebar tab switching for static published site.
 * Toggles between "Browse" (categories tree) and "Recent" (recent entries) views.
 */
(function() {
    'use strict';

    const STORAGE_KEY = 'memex-sidebar-tab';

    function setupSidebarTabs() {
        const tabs = document.querySelectorAll('.nav-tab');
        const treeSection = document.getElementById('tree-section');
        const recentSection = document.getElementById('recent-section');

        if (!tabs.length || !treeSection || !recentSection) {
            return;
        }

        // Restore saved tab preference
        const savedTab = localStorage.getItem(STORAGE_KEY) || 'tree';
        switchTab(savedTab);

        // Add click handlers
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;
                switchTab(tabName);
                localStorage.setItem(STORAGE_KEY, tabName);
            });
        });

        function switchTab(tabName) {
            // Update tab buttons
            tabs.forEach(t => t.classList.remove('active'));
            const activeTab = document.querySelector(`.nav-tab[data-tab="${tabName}"]`);
            if (activeTab) {
                activeTab.classList.add('active');
            }

            // Show/hide sections
            treeSection.style.display = tabName === 'tree' ? 'block' : 'none';
            recentSection.style.display = tabName === 'recent' ? 'block' : 'none';
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupSidebarTabs);
    } else {
        setupSidebarTabs();
    }
})();
