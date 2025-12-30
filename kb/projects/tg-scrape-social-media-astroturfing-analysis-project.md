---
title: TG Scrape: Social Media Astroturfing Analysis Project
tags:
  - tg-scrape
  - apify
  - facebook
  - astroturfing
  - sqlite
  - social-media-analysis
created: 2025-12-29
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
model: claude-opus-4-5
git_branch: main
last_edited_by: chris
---

# TG Scrape Project Documentation

## Overview
Investigative journalism project to analyze potential astroturfing in social media responses to political posts.

## Source Data
- **PDF**: "TG monitoring binder.pdf" (810 pages, ~10MB)
- Printed emails containing social media links
- Links were flagged by a politician's team for monitoring

## Extracted Links (stored in tg_scrape.db)
- **Total unique links**: 11,717
- By platform:
  - Twitter: 5,170 (mostly tweets)
  - YouTube: 3,184
  - Facebook: 1,862 (1,186 posts)
  - Reddit: 1,409
  - Instagram: 71
  - TikTok: 21

## Database Schema
Located at `/srv/fast/code/tg-scrape/schema.sql`

Key tables:
- `links` - Source URLs from PDF
- `posts` - Scraped post content
- `replies` - Comments/replies on posts
- `authors` - Unique commenters for tracking repeat actors
- `scrape_jobs` - Track Apify job status

## Apify Actors Used
| Platform | Actor | Cost |
|----------|-------|------|
| Facebook comments | `apify/facebook-comments-scraper` | $0.006/start + $0.0025/comment |
| Twitter replies | `scraper_one/x-post-replies-scraper` | $0.0025/start + $0.00025/reply |

## Scraping Strategy
- **Budget**: $40 Apify credits
- **Chosen approach**: Broad coverage - all 1,186 FB posts, 10 comments each (~$30)
- Posts are batched (50 per run) to minimize start costs

## Key Files
- `extract_links.py` - PDF to SQLite link extraction
- `schema.sql` - Database schema
- `db.py` - Database utilities
- `scraper.py` - Apify integration and cost estimation
- `store_results.py` - Store scraped data in SQLite

## Analysis Goals
1. Find repeat actors (same profileId across multiple posts)
2. Timing pattern analysis (coordinated responses)
3. Content similarity detection
4. Network analysis of reply patterns

## Status
- [x] PDF link extraction complete
- [x] SQLite schema created
- [x] Apify scraping tested and working
- [ ] Full Facebook scrape (1,186 posts)
- [ ] Twitter/X scraping
- [ ] Analysis queries

## Commands
```bash
# Check costs
python3 scraper.py estimate

# View database stats
python3 scraper.py stats

# Extract links from PDF
python3 extract_links.py
```
