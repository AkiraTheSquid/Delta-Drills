/* ================================================================
   SPLIT-TOOL.JS — PDF split job: form submit, polling, chapters
   ================================================================ */

const jobForm = document.getElementById("job-form");
const jobMessage = document.getElementById("job-message");
const jobStatus = document.getElementById("job-status");
const jobStatusText = document.getElementById("job-status-text");
const jobIdLabel = document.getElementById("job-id");
const chaptersSection = document.getElementById("chapters-section");
const chaptersList = document.getElementById("chapters-list");
const autoToc = document.getElementById("auto-toc");
const pageOffsetField = document.getElementById("page-offset-field");

const updateAutoTocState = () => {
  const isAuto = autoToc.checked;
  pageOffsetField.classList.toggle("hidden", isAuto);
};

autoToc.addEventListener("change", updateAutoTocState);
updateAutoTocState();

const renderChapters = (jobId, chapters) => {
  chaptersList.innerHTML = "";
  if (!chapters.length) {
    chaptersList.textContent = "No chapters generated.";
    return;
  }
  chapters.forEach((chapter) => {
    const row = document.createElement("div");
    row.className = "result-item";

    const title = document.createElement("span");
    title.textContent = `${chapter.title} (${chapter.start_page}-${chapter.end_page})`;

    const button = document.createElement("button");
    button.className = "ghost";
    button.type = "button";
    button.textContent = "Download";
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const response = await apiFetch(`/jobs/${jobId}/chapters/${chapter.id}/download`);
        if (!response.ok) {
          jobMessage.textContent = "Download failed.";
          return;
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = chapter.filename || "chapter.pdf";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        jobMessage.textContent = e.message;
      } finally {
        button.disabled = false;
      }
    });

    row.appendChild(title);
    row.appendChild(button);
    chaptersList.appendChild(row);
  });
};

const pollJob = async (jobId) => {
  try {
    const response = await apiFetch(`/jobs/${jobId}`);
    const data = await response.json();
    if (!response.ok) {
      jobStatusText.textContent = data.detail || "Failed to fetch job.";
      return;
    }
    jobStatusText.textContent = data.status;
    if (data.status === "completed") {
      const chaptersResponse = await apiFetch(`/jobs/${jobId}/chapters`);
      const chapters = await chaptersResponse.json();
      chaptersSection.classList.remove("hidden");
      renderChapters(jobId, chaptersResponse.ok ? chapters : []);
      return;
    }
    if (data.status === "failed") {
      jobMessage.textContent = data.error_message || "Job failed.";
      return;
    }
    setTimeout(() => pollJob(jobId), 2500);
  } catch (e) {
    jobStatusText.textContent = e.message;
  }
};

jobForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  jobMessage.textContent = "";
  chaptersSection.classList.add("hidden");

  const pdfFile = document.getElementById("pdf-file").files[0];
  const csvFile = document.getElementById("chapters-csv").files[0];
  const pageOffset = document.getElementById("page-offset").value || "0";

  if (!pdfFile) {
    jobMessage.textContent = "Please select a PDF.";
    return;
  }

  const formData = new FormData();
  formData.append("pdf_file", pdfFile);
  if (csvFile) {
    formData.append("chapters_csv", csvFile);
  }
  formData.append("auto_toc", autoToc.checked ? "true" : "false");
  formData.append("page_offset", pageOffset);
  const savedOpenaiKey = localStorage.getItem("account_openai_key") || "";
  const savedMathpixId = localStorage.getItem("account_mathpix_id") || "";
  const savedMathpixKey = localStorage.getItem("account_mathpix_key") || "";
  if (savedOpenaiKey) {
    formData.append("openai_api_key", savedOpenaiKey);
  }
  if (savedMathpixId) {
    formData.append("mathpix_app_id", savedMathpixId);
  }
  if (savedMathpixKey) {
    formData.append("mathpix_app_key", savedMathpixKey);
  }

  try {
    jobMessage.textContent = "Uploading...";
    const response = await apiFetch("/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      jobMessage.textContent = data.detail || "Job creation failed.";
      return;
    }
    jobMessage.textContent = "Job started.";
    jobStatus.classList.remove("hidden");
    jobIdLabel.textContent = data.id;
    jobStatusText.textContent = data.status;
    pollJob(data.id);
  } catch (e) {
    jobMessage.textContent = e.message;
  }
});
