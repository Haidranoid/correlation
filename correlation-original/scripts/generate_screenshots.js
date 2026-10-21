#!/usr/bin/env node

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:3000';
const SCREENSHOTS_DIR = path.join(__dirname, '..', 'screenshots');

const screenshots = [
  {
    name: 'home-page-full',
    url: '/',
    description: 'Complete home page showing dashboard, charts, and stock database table',
    viewport: { width: 1920, height: 1080 },
    waitFor: '.chart-container, #stocks-table_wrapper',
    delay: 3000, // Wait for all content to load
    fullPage: true // Capture entire page length
  }
];

// Stock detail page screenshot - check for available symbols
const conditionalScreenshots = [
  {
    name: 'stock-detail-full',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM'],
    description: 'stock detail page showing company information and price charts',
    viewport: { width: 1920, height: 1080 },
    waitFor: '.stock-info, .chart-container',
    delay: 3000,
    fullPage: true
  }
];

class ScreenshotGenerator {
  constructor() {
    this.browser = null;
    this.page = null;
  }

  async initialize() {
    console.log('🚀 Initializing Puppeteer...');
    
    this.browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--single-process',
        '--disable-gpu'
      ]
    });

    this.page = await this.browser.newPage();
    
    // Set default timeout
    this.page.setDefaultTimeout(30000);
    
    console.log('✅ Puppeteer initialized');
  }

  async checkServerRunning() {
    console.log('🔍 Checking if Rails server is running...');
    
    try {
      const response = await this.page.goto(BASE_URL, { waitUntil: 'domcontentloaded', timeout: 5000 });
      if (response.ok()) {
        console.log('✅ Rails server is running');
        return true;
      } else {
        console.log(`❌ Server responded with status ${response.status()}`);
        return false;
      }
    } catch (error) {
      console.log('❌ Rails server is not running');
      console.log('   Please start the server with: rails server');
      return false;
    }
  }

  async ensureScreenshotsDirectory() {
    if (!fs.existsSync(SCREENSHOTS_DIR)) {
      fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
      console.log(`📁 Created screenshots directory: ${SCREENSHOTS_DIR}`);
    }
  }

  async checkStockExists(symbol) {
    try {
      const response = await this.page.goto(`${BASE_URL}/stocks/${symbol}`, { 
        waitUntil: 'domcontentloaded', 
        timeout: 10000 
      });
      return response.ok() && !response.url().includes('404');
    } catch (error) {
      return false;
    }
  }

  async captureScreenshot(config) {
    const { name, url, description, viewport, waitFor, scrollTo, hover, delay, fullPage, focusOnTable } = config;
    const filename = `${name}.png`;
    const filepath = path.join(SCREENSHOTS_DIR, filename);

    console.log(`📸 Capturing: ${name}`);
    console.log(`   URL: ${BASE_URL}${url}`);
    console.log(`   Description: ${description}`);

    try {
      // Set viewport
      await this.page.setViewport(viewport);

      // Navigate to page
      const response = await this.page.goto(`${BASE_URL}${url}`, {
        waitUntil: 'networkidle0',
        timeout: 30000
      });

      if (!response.ok()) {
        console.log(`   ❌ Page returned status ${response.status()}`);
        return false;
      }

      // Wait for specific elements
      if (waitFor) {
        try {
          // Try multiple selectors separated by commas
          const selectors = waitFor.split(', ');
          let found = false;
          
          for (const selector of selectors) {
            try {
              await this.page.waitForSelector(selector.trim(), { timeout: 5000 });
              found = true;
              break;
            } catch (e) {
              // Try next selector
            }
          }
          
          if (!found) {
            console.log(`   ⚠️  Warning: Could not find any selector from "${waitFor}"`);
          }
        } catch (error) {
          console.log(`   ⚠️  Warning: Could not find selector "${waitFor}"`);
          // Continue anyway - might still get a useful screenshot
        }
      }

      // Additional wait for DataTables if this is the stocks table
      if (name === 'stocks-table') {
        try {
          // Wait for DataTables to be initialized and populated
          await this.page.waitForFunction(() => {
            const table = document.querySelector('#stocks-table');
            const wrapper = document.querySelector('#stocks-table_wrapper');
            return table && wrapper && table.rows.length > 1; // More than just header
          }, { timeout: 15000 });
          console.log('   ✅ DataTables loaded with data');
        } catch (error) {
          console.log('   ⚠️  Warning: DataTables may not be fully loaded');
        }
      }

      // Scroll to element if specified
      if (scrollTo) {
        try {
          if (focusOnTable) {
            // For table screenshots, scroll to bring table to top of viewport
            await this.page.$eval(scrollTo, el => el.scrollIntoView({ behavior: 'smooth', block: 'start' }));
            await this.page.waitForTimeout(1500); // Wait for scroll animation
            
            // Additional scroll up to ensure table header is visible
            await this.page.evaluate(() => {
              window.scrollBy(0, -100); // Scroll up a bit to show table header clearly
            });
            await this.page.waitForTimeout(500);
            console.log('   ✅ Table scrolled to optimal position');
          } else {
            await this.page.$eval(scrollTo, el => el.scrollIntoView({ behavior: 'smooth', block: 'center' }));
            await this.page.waitForTimeout(1000); // Wait for scroll animation
          }
        } catch (error) {
          console.log(`   ⚠️  Warning: Could not scroll to "${scrollTo}"`);
        }
      }

      // Hover over element if specified
      if (hover) {
        try {
          await this.page.hover(hover);
          if (delay) {
            await this.page.waitForTimeout(delay);
          }
        } catch (error) {
          console.log(`   ⚠️  Warning: Could not hover over "${hover}"`);
        }
      }

      // Take screenshot
      await this.page.screenshot({
        path: filepath,
        fullPage: fullPage || false,
        type: 'png'
      });

      // Check file size and optimize if needed
      const stats = fs.statSync(filepath);
      const fileSizeKB = Math.round(stats.size / 1024);
      console.log(`   📊 File size: ${fileSizeKB}KB`);

      if (fileSizeKB > 500) {
        console.log(`   🔧 Large file detected (${fileSizeKB}KB)`);
        console.log('   💡 Consider installing ImageMagick for automatic optimization');
      }

      console.log(`   ✅ Success: ${filename}`);
      return true;

    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      return false;
    }
  }

  async generateAll() {
    console.log('📸 Stock Correlation Analysis Platform - Screenshot Generator');
    console.log('=' * 60);
    console.log('');

    await this.initialize();
    
    if (!await this.checkServerRunning()) {
      await this.cleanup();
      process.exit(1);
    }

    await this.ensureScreenshotsDirectory();

    let successCount = 0;
    let totalCount = 0;

    // Generate main screenshots
    for (const config of screenshots) {
      totalCount++;
      if (await this.captureScreenshot(config)) {
        successCount++;
      }
      console.log('');
      
      // Brief pause between screenshots
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Generate conditional screenshots (check for any available stock symbols)
    console.log('🔍 Checking for available stock detail pages...');
    for (const config of conditionalScreenshots) {
      let foundSymbol = null;
      
      // Try each symbol until we find one that exists
      for (const symbol of config.symbols) {
        if (await this.checkStockExists(symbol)) {
          foundSymbol = symbol;
          break;
        }
      }
      
      if (foundSymbol) {
        console.log(`✅ Found ${foundSymbol}, capturing stock detail screenshot...`);
        
        // Create a screenshot config for the found symbol
        const stockConfig = {
          name: config.name,
          url: `/stocks/${foundSymbol}`,
          description: `${foundSymbol} ${config.description}`,
          viewport: config.viewport,
          waitFor: config.waitFor,
          fullPage: config.fullPage
        };
        
        totalCount++;
        if (await this.captureScreenshot(stockConfig)) {
          successCount++;
        }
        break; // Only capture one stock detail page
      } else {
        console.log(`⚠️  No stock symbols found from ${config.symbols.join(', ')}, skipping stock detail screenshot`);
      }
      console.log('');
    }

    console.log('🎉 Screenshot generation complete!');
    console.log(`📊 Results: ${successCount}/${totalCount} screenshots captured successfully`);
    console.log(`📁 Screenshots saved to: ${SCREENSHOTS_DIR}`);
    console.log('');
    console.log('💡 Next steps:');
    console.log('   1. Review screenshots in the screenshots/ directory');
    console.log('   2. Update documentation to reference the images');
    console.log('   3. Commit screenshots to version control');
    console.log('   4. Consider optimizing large images if needed');

    await this.cleanup();
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
      console.log('🧹 Cleaned up Puppeteer resources');
    }
  }
}

// Error handling
process.on('unhandledRejection', async (error) => {
  console.error('❌ Unhandled error:', error);
  process.exit(1);
});

process.on('SIGINT', async () => {
  console.log('\n⚠️  Interrupted by user');
  process.exit(0);
});

// Main execution
if (require.main === module) {
  const generator = new ScreenshotGenerator();
  
  generator.generateAll().catch(async (error) => {
    console.error('❌ Fatal error:', error);
    await generator.cleanup();
    process.exit(1);
  });
}