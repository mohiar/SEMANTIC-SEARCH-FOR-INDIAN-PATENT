function loadJSON() {
  // Directly specify the JSON file with the scholar search results.
  const fileName = "scholar_papers.json";

  fetch(fileName)
    .then(response => {
      if (!response.ok) {
        throw new Error(`File not found: ${fileName}`);
      }
      return response.json();
    })
    .then(data => {
      let outputDiv = document.getElementById("output");
      if (data.length === 0) {
        outputDiv.innerHTML = `<p>No relevant results found in ${fileName}.</p>`;
        return;
      }

      let html = "";
      data.forEach(item => {
        // Use the correct field names from your JSON: Title, URL, Abstract, Similarity
        html += `
          <div style="border: 1px solid #ccc; padding: 16px; margin-bottom: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0;">${item.title || 'No Title Available'}</h3>
            <p><strong>URL:</strong> <a href="${item.paper_url}" target="_blank" rel="noopener noreferrer">${item.paper_url}</a></p>
            <p><strong>Abstract:</strong> ${item.abstract || 'No abstract available.'}</p>
          </div>
        `;
      });

      outputDiv.innerHTML = html;
    })
    .catch(error => {
      document.getElementById("output").textContent = `Error: ${error.message}. Please ensure the file exists and is accessible.`;
      console.error('Error fetching JSON:', error);
    });
}