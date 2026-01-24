/**
 * Sidebar tab switching and mobile navigation for Memex static site.
 * - Toggles between "Browse" (categories tree) and "Recent" (recent entries) views
 * - Handles mobile menu open/close
 * - Handles mobile search overlay
 */
(function() {
    'use strict';

    const STORAGE_KEY = 'memex-sidebar-tab';

    // ===== SIDEBAR TABS =====
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

    // ===== MOBILE MENU =====
    function setupMobileMenu() {
        const menuBtn = document.querySelector('.mobile-menu-btn');
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.querySelector('.sidebar-overlay');

        if (!menuBtn || !sidebar) return;

        function openMenu() {
            menuBtn.classList.add('active');
            sidebar.classList.add('open');
            if (overlay) overlay.classList.add('active');
            document.body.classList.add('menu-open');
        }

        function closeMenu() {
            menuBtn.classList.remove('active');
            sidebar.classList.remove('open');
            if (overlay) overlay.classList.remove('active');
            document.body.classList.remove('menu-open');
        }

        function toggleMenu() {
            if (sidebar.classList.contains('open')) {
                closeMenu();
            } else {
                openMenu();
            }
        }

        // Menu button click
        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleMenu();
        });

        // Close on overlay click
        if (overlay) {
            overlay.addEventListener('click', closeMenu);
        }

        // Close on sidebar link click (for navigation)
        sidebar.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                // Small delay to let the navigation start
                setTimeout(closeMenu, 100);
            });
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && sidebar.classList.contains('open')) {
                closeMenu();
            }
        });

        // Close menu on resize to desktop
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                if (window.innerWidth > 768 && sidebar.classList.contains('open')) {
                    closeMenu();
                }
            }, 100);
        });
    }

    // ===== MOBILE SEARCH =====
    function setupMobileSearch() {
        const searchBtn = document.querySelector('.mobile-search-btn');
        const searchOverlay = document.querySelector('.search-overlay');
        const searchClose = document.querySelector('.search-overlay-close');
        const searchInput = document.querySelector('.search-overlay-input');
        const searchResults = document.querySelector('.search-overlay-results');
        const baseUrl = window.BASE_URL || '';

        if (!searchBtn || !searchOverlay) return;

        let searchIndex = null;
        let searchMetadata = null;
        let debounceTimer = null;

        function openSearch() {
            searchOverlay.classList.add('active');
            document.body.classList.add('menu-open');
            // Focus input after animation
            setTimeout(() => {
                if (searchInput) searchInput.focus();
            }, 100);
        }

        function closeSearch() {
            searchOverlay.classList.remove('active');
            document.body.classList.remove('menu-open');
            if (searchInput) searchInput.value = '';
            if (searchResults) searchResults.innerHTML = '';
        }

        // Open search
        searchBtn.addEventListener('click', openSearch);

        // Close search
        if (searchClose) {
            searchClose.addEventListener('click', closeSearch);
        }

        // Close on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && searchOverlay.classList.contains('active')) {
                closeSearch();
            }
        });

        // Load search index (reuse from main search if available)
        if (searchInput && searchResults) {
            fetch(baseUrl + '/search-index.json')
                .then(res => res.ok ? res.json() : Promise.reject('Failed to load'))
                .then(data => {
                    searchMetadata = data.metadata;
                    searchIndex = lunr(function() {
                        this.ref('id');
                        this.field('title', { boost: 10 });
                        this.field('tags', { boost: 5 });
                        this.field('content');
                        data.documents.forEach(function(doc) {
                            this.add(doc);
                        }, this);
                    });
                })
                .catch(err => console.error('Mobile search index error:', err));

            // Search input handler
            searchInput.addEventListener('input', () => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => performMobileSearch(), 150);
            });
        }

        function performMobileSearch() {
            const query = searchInput.value.trim();

            if (!query || !searchIndex) {
                searchResults.innerHTML = '';
                return;
            }

            let results;
            try {
                results = searchIndex.search(query);
                if (results.length === 0) {
                    results = searchIndex.search(query + '*');
                }
            } catch (e) {
                results = [];
            }

            if (results.length === 0) {
                searchResults.innerHTML = '<div class="search-result">No results found</div>';
            } else {
                searchResults.innerHTML = results.slice(0, 15).map(result => {
                    const meta = searchMetadata[result.ref];
                    if (!meta) return '';

                    const tagsHtml = meta.tags && meta.tags.length > 0
                        ? '<div class="search-tags">' + meta.tags.join(', ') + '</div>'
                        : '';

                    return '<div class="search-result">' +
                        '<a href="' + baseUrl + '/' + meta.path + '">' + escapeHtml(meta.title) + '</a>' +
                        tagsHtml +
                        '</div>';
                }).join('');
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    }

    // ===== BOTTOM NAV SEARCH BUTTON =====
    function setupBottomNavSearch() {
        const bottomSearchBtn = document.querySelector('.bottom-nav-search');
        const searchOverlay = document.querySelector('.search-overlay');
        const searchInput = document.querySelector('.search-overlay-input');

        if (!bottomSearchBtn || !searchOverlay) return;

        bottomSearchBtn.addEventListener('click', (e) => {
            e.preventDefault();
            searchOverlay.classList.add('active');
            document.body.classList.add('menu-open');
            setTimeout(() => {
                if (searchInput) searchInput.focus();
            }, 100);
        });
    }

    // ===== INITIALIZE =====
    function init() {
        setupSidebarTabs();
        setupMobileMenu();
        setupMobileSearch();
        setupBottomNavSearch();
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
