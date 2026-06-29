(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutPanelUtils = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function escapeHtml(value) {
        if (value === undefined || value === null) return "";
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function escapeJsxDoubleQuotedString(value) {
        if (!value) return "";
        return String(value)
            .replace(/\\/g, "\\\\")
            .replace(/"/g, '\\"')
            .replace(/\n/g, "\\n")
            .replace(/\r/g, "\\r")
            .replace(/\t/g, "\\t");
    }

    function createLazyDomProxy(documentRef, cache) {
        var doc = documentRef || (typeof document !== "undefined" ? document : null);
        var target = cache || {};
        return new Proxy(target, {
            get: function (targetObj, id) {
                if (typeof id !== "string") return targetObj[id];
                if (id in targetObj) return targetObj[id];
                var node = doc && typeof doc.getElementById === "function"
                    ? doc.getElementById(id)
                    : null;
                if (node) targetObj[id] = node;
                return node;
            }
        });
    }

    function normalizePaletteText(value) {
        return (value || "").toLowerCase().replace(/\s+/g, " ").trim();
    }

    function formatPaletteLabel(value) {
        return (value || "").replace(/[-_]+/g, " ").replace(/\b[a-z]/g, function (letter) {
            return letter.toUpperCase();
        });
    }

    function _copyBooleanSetting(source, target, key) {
        if (typeof source[key] === "boolean") target[key] = source[key];
    }

    function _copyEnumSetting(source, target, key, allowed) {
        if (typeof source[key] !== "string") return;
        if (allowed.indexOf(source[key]) === -1) return;
        target[key] = source[key];
    }

    function normalizeLocalSettings(value) {
        var source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
        var settings = {};
        _copyBooleanSetting(source, settings, "autoImport");
        _copyBooleanSetting(source, settings, "autoOpen");
        _copyBooleanSetting(source, settings, "showNotifications");
        _copyEnumSetting(source, settings, "outputDir", ["source", "project", "custom"]);
        _copyEnumSetting(source, settings, "theme", ["auto", "dark", "light"]);
        _copyEnumSetting(source, settings, "lang", ["en"]);
        _copyEnumSetting(source, settings, "defaultModel", [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v1", "large-v2",
            "large-v3", "turbo", "large-v3-turbo", "distil-large-v2",
            "distil-large-v3", "distil-large-v3.5", "distil-medium.en",
            "distil-small.en",
        ]);
        return settings;
    }

    function getCommandPaletteItemKey(item) {
        if (!item) return "";
        return [item.name || "", item.tab || "", item.sub || ""].join("::");
    }

    var DEFAULT_TAB_DESCRIPTIONS = {
        cut: "Tighten pacing, trims, and spoken edits from one focused cut workflow.",
        captions: "Transcribe, translate, and shape subtitle deliverables without leaving the panel.",
        audio: "Polish dialogue, stems, loudness, and generated sound from one audio surface.",
        video: "Repair, reframe, and finish image work with cleaner visual controls.",
        export: "Build deliverables, thumbnails, and repeatable output presets faster.",
        timeline: "Write sequence edits and timeline metadata back into Premiere with more control.",
        nlp: "Use search and language-driven tools to find footage or trigger edit actions.",
        settings: "Adjust workspace defaults, templates, and system-level behavior.",
        _default: "Open this tool and jump directly to the matching workspace.",
        _none: "Open tools across the editing workflow.",
    };

    function descriptionForItem(item, descriptionMap) {
        var itemKey = getCommandPaletteItemKey(item);
        if (descriptionMap && descriptionMap[itemKey]) return descriptionMap[itemKey];
        if (!item) return (descriptionMap && descriptionMap._none) || DEFAULT_TAB_DESCRIPTIONS._none;
        var tabDescs = (descriptionMap && descriptionMap._tabDescriptions) || DEFAULT_TAB_DESCRIPTIONS;
        return tabDescs[item.tab] || tabDescs._default || DEFAULT_TAB_DESCRIPTIONS._default;
    }

    var DEFAULT_SECTION_LABELS = {
        recent: "Recent",
        favorites: "Favorites",
        currentWorkspace: "Current Workspace",
        suggestedTools: "Suggested Tools",
        browseAll: "Browse All",
        matchingTools: "Matching Tools",
    };

    function makePaletteContext(options) {
        options = options || {};
        return {
            sectionLabels: options.sectionLabels || DEFAULT_SECTION_LABELS,
            items: Array.isArray(options.items) ? options.items : [],
            query: normalizePaletteText(options.query),
            activeTab: options.activeTab || "",
            historyKeys: Array.isArray(options.historyKeys) ? options.historyKeys : [],
            favoriteIds: Array.isArray(options.favoriteIds) ? options.favoriteIds : [],
            descriptionMap: options.descriptionMap || {},
            getTabLabel: typeof options.getTabLabel === "function"
                ? options.getTabLabel
                : function (tab) { return formatPaletteLabel(tab); },
            getSubLabel: typeof options.getSubLabel === "function"
                ? options.getSubLabel
                : function (sub) { return formatPaletteLabel(sub); },
            getFavoriteId: typeof options.getFavoriteId === "function"
                ? options.getFavoriteId
                : function () { return ""; },
            getItemForFavorite: typeof options.getItemForFavorite === "function"
                ? options.getItemForFavorite
                : function () { return null; }
        };
    }

    function createPaletteEntry(item, ctx, extras) {
        extras = extras || {};
        var key = getCommandPaletteItemKey(item);
        var favoriteId = ctx.getFavoriteId(item) || "";
        var tabLabel = ctx.getTabLabel(item.tab);
        var subLabel = ctx.getSubLabel(item.sub);
        return {
            item: item,
            key: key,
            description: descriptionForItem(item, ctx.descriptionMap),
            tabLabel: tabLabel,
            subLabel: subLabel,
            location: subLabel ? (tabLabel + " / " + subLabel) : tabLabel,
            favoriteId: favoriteId,
            isFavorite: favoriteId ? ctx.favoriteIds.indexOf(favoriteId) !== -1 : false,
            isRecent: !!extras.isRecent,
            isCurrent: !!extras.isCurrent,
            isRunnable: item.runnable !== false,
            readiness: item.readiness || "",
            routeValid: item.route_valid !== false,
            readinessReason: item.readiness_reason || "",
            score: extras.score || 0
        };
    }

    function scoreCommandPaletteItem(item, ctx) {
        var q = ctx.query;
        if (!q) return 0;

        var name = normalizePaletteText(item.name);
        var keywords = normalizePaletteText(item.keywords);
        var tabLabel = normalizePaletteText(ctx.getTabLabel(item.tab));
        var subLabel = normalizePaletteText(ctx.getSubLabel(item.sub));
        var score = 0;
        var matchedTokens = 0;
        var tokens = q.split(" ");
        var favoriteId = ctx.getFavoriteId(item);

        if (name === q) score += 220;
        else if (name.indexOf(q) === 0) score += 140;
        else if (name.indexOf(q) !== -1) score += 96;

        if (keywords.indexOf(q) !== -1) score += 56;
        if (tabLabel.indexOf(q) !== -1) score += 24;
        if (subLabel.indexOf(q) !== -1) score += 28;

        for (var i = 0; i < tokens.length; i++) {
            var token = tokens[i];
            if (!token) continue;
            if (name.indexOf(token) !== -1) {
                score += 32;
                matchedTokens++;
            } else if (keywords.indexOf(token) !== -1) {
                score += 18;
                matchedTokens++;
            } else if (tabLabel.indexOf(token) !== -1 || subLabel.indexOf(token) !== -1) {
                score += 10;
                matchedTokens++;
            }
        }

        if (!score && !matchedTokens) return 0;
        if (matchedTokens > 1) score += matchedTokens * 8;
        if (favoriteId && ctx.favoriteIds.indexOf(favoriteId) !== -1) score += 16;
        if (item.tab === ctx.activeTab) score += 12;
        return score;
    }

    function getItemByKey(items, key) {
        for (var i = 0; i < items.length; i++) {
            if (getCommandPaletteItemKey(items[i]) === key) return items[i];
        }
        return null;
    }

    function buildPaletteEntries(items, ctx, resolver, seen) {
        var entries = [];
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            if (!item) continue;
            var key = getCommandPaletteItemKey(item);
            if (seen && seen[key]) continue;
            if (seen) seen[key] = true;
            entries.push(createPaletteEntry(item, ctx, resolver ? resolver(item, key, i) : null));
        }
        return entries;
    }

    function addPaletteSection(sections, label, items, ctx, resolver, seen) {
        var entries = buildPaletteEntries(items, ctx, resolver, seen);
        if (entries.length) sections.push({ label: label, entries: entries });
    }

    function buildCommandPaletteSections(options) {
        var ctx = makePaletteContext(options);
        var sections = [];
        var historyLookup = {};
        for (var i = 0; i < ctx.historyKeys.length; i++) historyLookup[ctx.historyKeys[i]] = true;

        if (!ctx.query) {
            var seen = {};
            var recentItems = [];
            for (i = 0; i < ctx.historyKeys.length; i++) {
                var historyItem = getItemByKey(ctx.items, ctx.historyKeys[i]);
                if (historyItem) recentItems.push(historyItem);
            }

            var favoriteItems = [];
            for (i = 0; i < ctx.favoriteIds.length; i++) {
                favoriteItems.push(ctx.getItemForFavorite(ctx.favoriteIds[i]));
            }

            var currentItems = [];
            for (i = 0; i < ctx.items.length; i++) {
                if (ctx.items[i].tab === ctx.activeTab) currentItems.push(ctx.items[i]);
            }

            var browseItems = ctx.items.slice(0);
            browseItems.sort(function (a, b) {
                var tabCompare = ctx.getTabLabel(a.tab).localeCompare(ctx.getTabLabel(b.tab));
                if (tabCompare !== 0) return tabCompare;
                return (a.name || "").localeCompare(b.name || "");
            });

            var sl = ctx.sectionLabels || DEFAULT_SECTION_LABELS;
            addPaletteSection(sections, sl.recent, recentItems, ctx, function (item) {
                return { isRecent: true, isCurrent: item.tab === ctx.activeTab };
            }, seen);

            addPaletteSection(sections, sl.favorites, favoriteItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: item.tab === ctx.activeTab };
            }, seen);

            addPaletteSection(sections, ctx.activeTab ? sl.currentWorkspace : sl.suggestedTools, currentItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: true };
            }, seen);

            addPaletteSection(sections, sl.browseAll, browseItems, ctx, function (item, key) {
                return { isRecent: !!historyLookup[key], isCurrent: item.tab === ctx.activeTab };
            }, seen);
            return sections;
        }

        var matches = [];
        for (i = 0; i < ctx.items.length; i++) {
            var item = ctx.items[i];
            var score = scoreCommandPaletteItem(item, ctx);
            if (!score) continue;
            matches.push(createPaletteEntry(item, ctx, {
                score: score,
                isRecent: !!historyLookup[getCommandPaletteItemKey(item)],
                isCurrent: item.tab === ctx.activeTab
            }));
        }

        matches.sort(function (a, b) {
            if (b.score !== a.score) return b.score - a.score;
            if (a.isFavorite !== b.isFavorite) return a.isFavorite ? -1 : 1;
            if (a.isCurrent !== b.isCurrent) return a.isCurrent ? -1 : 1;
            return (a.item.name || "").localeCompare(b.item.name || "");
        });

        if (matches.length) sections.push({ label: (ctx.sectionLabels || DEFAULT_SECTION_LABELS).matchingTools, entries: matches });
        return sections;
    }

    return {
        escapeHtml: escapeHtml,
        escapeJsxDoubleQuotedString: escapeJsxDoubleQuotedString,
        createLazyDomProxy: createLazyDomProxy,
        normalizePaletteText: normalizePaletteText,
        normalizeLocalSettings: normalizeLocalSettings,
        formatPaletteLabel: formatPaletteLabel,
        getCommandPaletteItemKey: getCommandPaletteItemKey,
        scoreCommandPaletteItem: function (item, options) {
            return scoreCommandPaletteItem(item, makePaletteContext(options));
        },
        buildCommandPaletteSections: buildCommandPaletteSections
    };
});
