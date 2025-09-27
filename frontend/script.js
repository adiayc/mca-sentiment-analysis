// Configuration
const FLASK_API_URL = 'http://localhost:5000';

// Set your X API Bearer Token here
const X_API_BEARER_TOKEN = 'X_API_BEARER_TOKEN';

// Extract post ID from X.com or Twitter.com URL
function extractPostId(url) {
    // Support both x.com and twitter.com domains
    const regex = /(?:twitter\.com|x\.com)\/\w+\/status\/(\d+)/;
    const match = url.match(regex);
    return match ? match[1] : null;
}

// Fetch replies using Flask backend proxy
async function fetchPostReplies(postId, bearerToken) {
    try {
        const response = await fetch(`${FLASK_API_URL}/fetch_replies`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                post_id: postId,
                bearer_token: bearerToken
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`API Error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        return data.replies;
    } catch (error) {
        console.error('Error fetching replies:', error);
        throw error;
    }
}

// Send comments to Flask API for sentiment analysis
async function analyzeSentiment(comments) {
    try {
        const response = await fetch(`${FLASK_API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ comments: comments })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`API Error: ${errorData.error || response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error analyzing sentiment:', error);
        throw error;
    }
}

// Generate sentiment chart
async function generateChart(sentimentCounts) {
    try {
        const response = await fetch(`${FLASK_API_URL}/generate_chart`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sentiment_counts: sentimentCounts })
        });

        if (!response.ok) {
            throw new Error('Failed to generate chart');
        }

        return await response.blob();
    } catch (error) {
        console.error('Error generating chart:', error);
        throw error;
    }
}

// Generate word cloud
async function generateWordCloud(comments) {
    try {
        const response = await fetch(`${FLASK_API_URL}/generate_wordcloud`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ comments: comments })
        });

        if (!response.ok) {
            throw new Error('Failed to generate word cloud');
        }

        return await response.blob();
    } catch (error) {
        console.error('Error generating word cloud:', error);
        throw error;
    }
}

// Display results
function displayResults(analysisResults, chartBlob, wordcloudBlob) {
    // Calculate sentiment counts
    const sentimentCounts = { '1': 0, '0': 0, '-1': 0 };
    analysisResults.forEach(result => {
        sentimentCounts[result.sentiment]++;
    });

    // Update summary stats
    document.getElementById('totalReplies').textContent = analysisResults.length;
    document.getElementById('positiveCount').textContent = sentimentCounts['1'];
    document.getElementById('neutralCount').textContent = sentimentCounts['0'];
    document.getElementById('negativeCount').textContent = sentimentCounts['-1'];

    // Display chart and word cloud
    if (chartBlob) {
        const chartUrl = URL.createObjectURL(chartBlob);
        document.getElementById('chartImage').src = chartUrl;
    }

    if (wordcloudBlob) {
        const wordcloudUrl = URL.createObjectURL(wordcloudBlob);
        document.getElementById('wordcloudImage').src = wordcloudUrl;
    }

    // Display individual comments
    const commentsList = document.getElementById('commentsList');
    commentsList.innerHTML = '';

    analysisResults.forEach(result => {
        const commentDiv = document.createElement('div');
        const sentimentClass = result.sentiment === '1' ? 'positive' : 
                             result.sentiment === '-1' ? 'negative' : 'neutral';
        const sentimentLabel = result.sentiment === '1' ? 'Positive' : 
                              result.sentiment === '-1' ? 'Negative' : 'Neutral';

        commentDiv.className = `comment-item ${sentimentClass}`;
        commentDiv.innerHTML = `
            <div>
                ${result.comment}
                <span class="sentiment-badge ${sentimentClass}">${sentimentLabel}</span>
            </div>
        `;
        commentsList.appendChild(commentDiv);
    });

    // Show results
    document.getElementById('results').style.display = 'block';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Hide error message
function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Validate bearer token
function validateBearerToken() {
    if (!X_API_BEARER_TOKEN || X_API_BEARER_TOKEN === 'YOUR_BEARER_TOKEN_HERE') {
        showError('Please set your X API Bearer Token in the script.js file (X_API_BEARER_TOKEN variable)');
        return false;
    }
    return true;
}

// Main analysis function
async function analyzePost() {
    const postUrl = document.getElementById('postUrl').value.trim();

    if (!postUrl) {
        showError('Please enter a post URL');
        return;
    }

    if (!validateBearerToken()) {
        return;
    }

    const postId = extractPostId(postUrl);
    if (!postId) {
        showError('Invalid post URL format. Please use x.com or twitter.com URLs');
        return;
    }

    hideError();
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        // Step 1: Fetch replies via Flask proxy
        console.log('Fetching replies for post:', postId);
        const replies = await fetchPostReplies(postId, X_API_BEARER_TOKEN);
        console.log(`Found ${replies.length} replies`);

        // Step 2: Analyze sentiment
        console.log('Analyzing sentiment...');
        const analysisResults = await analyzeSentiment(replies);

        // Step 3: Calculate sentiment counts for chart
        const sentimentCounts = { '1': 0, '0': 0, '-1': 0 };
        analysisResults.forEach(result => {
            sentimentCounts[result.sentiment]++;
        });

        // Step 4: Generate visualizations
        console.log('Generating visualizations...');
        const [chartBlob, wordcloudBlob] = await Promise.all([
            generateChart(sentimentCounts),
            generateWordCloud(replies)
        ]);

        // Step 5: Display results
        displayResults(analysisResults, chartBlob, wordcloudBlob);

    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Legacy function name for backward compatibility
function analyzeTweet() {
    analyzePost();
}

// Event listeners for Enter key functionality
document.addEventListener('DOMContentLoaded', function() {
    // Allow Enter key to trigger analysis
    const postUrlInput = document.getElementById('postUrl') || document.getElementById('tweetUrl');
    if (postUrlInput) {
        postUrlInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzePost();
            }
        });
    }
});
