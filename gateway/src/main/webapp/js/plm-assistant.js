/*
 * Front-end of the PLM assistant: object-oriented JavaScript with jQuery.
 *   ApiClient      : talks to the Jakarta EE gateway (/api/*)
 *   ResultRenderer : renders filters, answer, sources and history (text only, no innerHTML)
 *   PlmAssistant   : wires the form, the client and the renderer together
 */
(function ($) {
  "use strict";

  class ApiClient {
    constructor(baseUrl) {
      this.baseUrl = baseUrl;
    }

    ask(question) {
      return $.ajax({
        url: this.baseUrl + "/ask",
        method: "POST",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({ question: question })
      });
    }

    history() {
      return $.getJSON(this.baseUrl + "/history");
    }
  }

  class ResultRenderer {
    constructor($root) {
      this.$status = $root.find("#status");
      this.$filters = $root.find("#filters");
      this.$answer = $root.find("#answer");
      this.$sources = $root.find("#sources");
      this.$history = $root.find("#history");
    }

    loading() {
      this.$status.text("Recherche en cours…").prop("hidden", false);
      this.$sources.empty();
      this.$answer.prop("hidden", true);
      this.$filters.prop("hidden", true);
    }

    error(message) {
      this.$status.text(message).prop("hidden", false);
    }

    result(data) {
      this.$status.prop("hidden", true);
      this.renderFilters(data.parsed.filters);
      this.$answer.text(data.answer).prop("hidden", false);
      data.sources.forEach((src) => this.$sources.append(this.sourceItem(src)));
    }

    renderFilters(filters) {
      const labels = {
        start: "Depuis", end: "Jusqu'au", status: "Statut",
        doc_type: "Type", part_number: "Pièce", assembly: "Assemblage"
      };
      this.$filters.empty();
      Object.keys(labels).forEach((key) => {
        if (filters[key]) {
          this.$filters.append($("<span>", { "class": "chip" }).text(labels[key] + " : " + filters[key]));
        }
      });
      this.$filters.prop("hidden", this.$filters.children().length === 0);
    }

    sourceItem(src) {
      return $("<li>")
        .append($("<div>", { "class": "meta" }).text(src.id + " · " + src.date + " · " + src.status))
        .append($("<div>", { "class": "text" }).text(src.text));
    }

    history(entries) {
      this.$history.empty();
      entries.forEach((e) => {
        this.$history.append($("<li>").text(e.question + " (" + e.resultCount + " résultats, " + e.latencyMs + " ms)"));
      });
    }
  }

  class PlmAssistant {
    constructor($root, client) {
      this.$form = $root.find("#ask-form");
      this.$input = $root.find("#question");
      this.client = client;
      this.renderer = new ResultRenderer($root);
    }

    init() {
      this.$form.on("submit", (event) => {
        event.preventDefault();
        this.submit(this.$input.val().trim());
      });
      this.refreshHistory();
      return this;
    }

    submit(question) {
      if (question.length < 3) {
        this.renderer.error("La question doit contenir au moins 3 caractères.");
        return;
      }
      this.renderer.loading();
      this.client.ask(question)
        .done((data) => this.renderer.result(data))
        .fail((xhr) => this.renderer.error((xhr.responseJSON && xhr.responseJSON.error) || "Erreur du service."))
        .always(() => this.refreshHistory());
    }

    refreshHistory() {
      this.client.history().done((entries) => this.renderer.history(entries));
    }
  }

  $(function () {
    new PlmAssistant($("main"), new ApiClient("api")).init();
  });
})(jQuery);
