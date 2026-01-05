/**
 * Lunr.js client-side search for Memex static site.
 *
 * Loads the search index and provides instant search functionality.
 */
(function() {
    'use strict';

    let searchIndex = null;
    let searchMetadata = null;
    let debounceTimer = null;

    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const baseUrl = window.BASE_URL || '';

    if (!searchInput || !searchResults) {
        console.warn('Search elements not found');
        return;
    }

    // Load search index
    fetch(baseUrl + '/search-index.json')
        .then(function(res) {
            if (!res.ok) throw new Error('Failed to load search index');
            return res.json();
        })
        .then(function(data) {
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
        .catch(function(err) {
            console.error('Search index load error:', err);
        });

    // Handle search input with debouncing
    searchInput.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(performSearch, 150);
    });

    function performSearch() {
        var query = searchInput.value.trim();

        if (!query || !searchIndex) {
            searchResults.classList.remove('active');
            searchResults.innerHTML = '';
            return;
        }

        var results;
        try {
            // Try exact search first, fall back to fuzzy
            results = searchIndex.search(query);
            if (results.length === 0) {
                results = searchIndex.search(query + '*');
            }
        } catch (e) {
            // Handle Lunr query syntax errors
            results = [];
        }

        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result">No results found</div>';
        } else {
            searchResults.innerHTML = results.slice(0, 10).map(function(result) {
                var meta = searchMetadata[result.ref];
                if (!meta) return '';

                var tagsHtml = meta.tags && meta.tags.length > 0
                    ? '<div class="search-tags">' + meta.tags.join(', ') + '</div>'
                    : '';

                return '<div class="search-result">' +
                    '<a href="' + baseUrl + '/' + meta.path + '">' + escapeHtml(meta.title) + '</a>' +
                    tagsHtml +
                    '</div>';
            }).join('');
        }

        searchResults.classList.add('active');
    }

    // Close results on click outside
    document.addEventListener('click', function(e) {
        if (!searchResults.contains(e.target) && e.target !== searchInput) {
            searchResults.classList.remove('active');
        }
    });

    // Close results on Escape
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            searchResults.classList.remove('active');
            searchInput.blur();
        }
    });

    // Helper to escape HTML
    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
})();
