---
title: TG Scrape - Social Media Astroturfing Analysis
tags:
  - project
  - tg-scrape
  - apify
  - facebook
  - scraping
  - astroturfing
  - sqlite
  - investigation
created: 2025-12-30
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
model: claude-opus-4-5
git_branch: main
last_edited_by: chris
---

# TG Scrape - Social Media Astroturfing Analysis

## Overview

Investigation project to analyze potential astroturfing patterns in social media responses to political posts. Source data is a ~10MB PDF ("TG monitoring binder.pdf") containing 810 pages of printed emails with social media links that were being monitored by a politician's team.

## Project Goals

1. Extract all social media links from the PDF
2. Store in SQLite database with proper schema
3. Use Apify to scrape posts and comments from platforms
4. Track post types and author patterns
5. Analyze for astroturfing indicators (repeat actors, timing patterns, coordinated behavior)

## Current Status

### Completed
- **PDF Extraction**: 11,717 unique links extracted (26,918 total, many duplicates)
- **Database Schema**: SQLite with tables for links, posts, replies, authors
- **Facebook Posts**: 1,023 of 1,186 posts scraped (86% success rate)
- **Test Comments**: 1,402 comments from early test batch

### Link Breakdown by Platform
| Platform | Links | Status |
|----------|------:|--------|
| Twitter/X | 5,170 | Not scraped |
| YouTube | 3,184 | Not scraped |
| Facebook | 1,862 | 1,023 posts scraped |
| Reddit | 1,409 | Not scraped |
| Instagram | 71 | Not scraped |
| TikTok | 21 | Not scraped |

### Facebook Link Types
- 1,186 posts (scraped)
- 491 videos/reels (unknown type)
- 75 profile/page links
- 67 group links
- 39 video links
- 4 photo links

## Technical Implementation

### Key Files
- `extract_links.py` - PDF text extraction with pypdf, URL classification
- `schema.sql` - SQLite schema with links, posts, replies, authors tables
- `db.py` - Database utilities and connection management
- `scraper.py` - Apify integration and cost estimation
- `store_results.py` - Store scraped data in SQLite
- `run_scrape_fast.py` - Parallel batch scraper (5 concurrent Apify runs)

### Database Schema Highlights
```sql
-- Key analysis view
CREATE VIEW frequent_commenters AS
SELECT platform, author_username, 
       COUNT(DISTINCT post_id) as posts_commented_on,
       COUNT(*) as total_comments
FROM replies WHERE author_username IS NOT NULL
GROUP BY platform, author_username 
HAVING COUNT(DISTINCT post_id) > 1
ORDER BY posts_commented_on DESC;
```

### Apify Integration
- Actor: `apify/facebook-comments-scraper`
- Pricing: $0.006/start + $0.0025/comment
- Parallel execution: 5 batches concurrent (~175 posts/min)
- API token stored in scripts (not committed)

## Next Steps

1. **Get Facebook comment counts** - Need to determine which posts have high engagement before bulk scraping
2. **Scrape Facebook comments** - Full comment scrape for astroturfing analysis (~$25 for 10k comments)
3. **Twitter/X scraping** - 5,170 links (largest dataset)
4. **Build analysis queries** - Identify repeat commenters, timing patterns, coordinated behavior

## Budget
- $40 Apify credits available
- Current spend: ~$5-8 (test batches + metadata scrape)
- Remaining: ~$32-35

## Resources
- Decodo residential proxy: 2GB available
- Decodo ISP static residential: 100GB (10 IPs)
