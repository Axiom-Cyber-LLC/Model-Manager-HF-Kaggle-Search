#!/usr/bin/env swift
import Cocoa

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 1080, height: 760),
        styleMask: [.titled, .closable, .miniaturizable, .resizable],
        backing: .buffered,
        defer: false
    )

    // Guided search/download controls.
    private let specificCheck = NSButton(checkboxWithTitle: "Download a single specific Hugging Face repo", target: nil, action: nil)
    private let specificRepoField = NSTextField()
    private let specificKindPopup = NSPopUpButton()
    private let searchTermsField = NSTextField()
    private let searchKindPopup = NSPopUpButton()
    private let searchSourcePopup = NSPopUpButton()
    private let sizeRangeField = NSTextField(string: "200 MB - 2 TB")
    private let authorsField = NSTextField(string: "DavidAU, TheBloke, mradermacher, bartowski")
    private let termsField = NSTextField(string: "uncensored, nsfw, roleplay, erotic, jailbreak")
    private let familyField = NSTextField()
    private let resultLimitField = NSTextField(string: "50")
    private let pageSizeField = NSTextField(string: "20")
    private let hideDuplicatesCheck = NSButton(checkboxWithTitle: "Hide likely duplicate/mirror repos", target: nil, action: nil)
    private let searchRootField = NSTextField()
    private let ggufCheck = NSButton(checkboxWithTitle: "GGUF", target: nil, action: nil)
    private let coreMLCheck = NSButton(checkboxWithTitle: "Core ML", target: nil, action: nil)
    private let mlxCheck = NSButton(checkboxWithTitle: "MLX", target: nil, action: nil)
    private let onnxCheck = NSButton(checkboxWithTitle: "ONNX", target: nil, action: nil)
    private let safetensorsCheck = NSButton(checkboxWithTitle: "Safetensors", target: nil, action: nil)
    private let kerasCheck = NSButton(checkboxWithTitle: "Keras/TensorFlow", target: nil, action: nil)
    private let pytorchCheck = NSButton(checkboxWithTitle: "Raw PyTorch", target: nil, action: nil)
    private let anyArtifactCheck = NSButton(checkboxWithTitle: "Any artifact type", target: nil, action: nil)

    // Direct hfdownloader controls.
    private let directRepoField = NSTextField()
    private let directRootField = NSTextField()
    private let directFilterField = NSTextField()
    private let directExcludeField = NSTextField()
    private let directEndpointField = NSTextField()
    private let connectionsField = NSTextField(string: "16")
    private let maxActiveField = NSTextField(string: "4")
    private let directTypePopup = NSPopUpButton()
    private let verifyPopup = NSPopUpButton()
    private let dryRunCheck = NSButton(checkboxWithTitle: "Dry run only", target: nil, action: nil)
    private let startButton = NSButton(title: "Start Direct Download", target: nil, action: nil)
    private let stopButton = NSButton(title: "Stop", target: nil, action: nil)
    private let outputView = NSTextView()

    // Audit, prep, and leaderboard controls.
    private let auditDryRunCheck = NSButton(checkboxWithTitle: "Audit dry run", target: nil, action: nil)
    private let prepareDryRunCheck = NSButton(checkboxWithTitle: "Prepare dry run", target: nil, action: nil)
    private let prepareWorkersField = NSTextField(string: "8")
    private let prepareOnlyField = NSTextField()
    private let prepareSkipField = NSTextField()
    private let prepareNoAuditCheck = NSButton(checkboxWithTitle: "Skip model_audit", target: nil, action: nil)
    private let prepareNonInteractiveAuditCheck = NSButton(checkboxWithTitle: "Non-interactive audit", target: nil, action: nil)
    private let prepareSkipDuplicatesAuditCheck = NSButton(checkboxWithTitle: "Skip duplicate audit", target: nil, action: nil)
    private let prepareContinueOnErrorCheck = NSButton(checkboxWithTitle: "Continue on error", target: nil, action: nil)
    private let conversionRootField = NSTextField()
    private let conversionQuantField = NSTextField(string: "Q4_K_M")
    private let conversionWorkersField = NSTextField(string: "1")
    private let conversionSelectField = NSTextField()
    private let conversionListOnlyCheck = NSButton(checkboxWithTitle: "List only", target: nil, action: nil)
    private let conversionDryRunCheck = NSButton(checkboxWithTitle: "Dry run", target: nil, action: nil)
    private let conversionNoAuditCheck = NSButton(checkboxWithTitle: "Skip audit", target: nil, action: nil)
    private let conversionNonInteractiveAuditCheck = NSButton(checkboxWithTitle: "Non-interactive audit", target: nil, action: nil)
    private let conversionYesPrepareCheck = NSButton(checkboxWithTitle: "Run prepare after conversion", target: nil, action: nil)
    private let conversionForceCheck = NSButton(checkboxWithTitle: "Force reconvert", target: nil, action: nil)
    private let generalBoardCheck = NSButton(checkboxWithTitle: "general", target: nil, action: nil)
    private let codingBoardCheck = NSButton(checkboxWithTitle: "coding", target: nil, action: nil)
    private let visualBoardCheck = NSButton(checkboxWithTitle: "visual", target: nil, action: nil)
    private let securityBoardCheck = NSButton(checkboxWithTitle: "security", target: nil, action: nil)
    private let embeddingBoardCheck = NSButton(checkboxWithTitle: "embedding", target: nil, action: nil)
    private let searchSpacesCheck = NSButton(checkboxWithTitle: "Search Hugging Face Spaces", target: nil, action: nil)
    private let manualCacheCheck = NSButton(checkboxWithTitle: "Manually add/update cached model IDs", target: nil, action: nil)
    private let refreshCacheCheck = NSButton(checkboxWithTitle: "Refresh cached leaderboard model IDs", target: nil, action: nil)
    private let leaderboardLimitField = NSTextField(string: "20")

    private var process: Process?

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildWindow()
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func buildWindow() {
        window.title = "Model Manager"
        let content = NSView()
        content.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = content

        let title = NSTextField(labelWithString: "Model Manager")
        title.font = NSFont.boldSystemFont(ofSize: 24)

        let subtitle = NSTextField(labelWithString: "Native macOS GUI for modelmgr. No web server, no localhost port, no Docker. Interactive result selection still happens in Terminal so the Python manager remains the source of truth.")
        subtitle.textColor = .secondaryLabelColor
        subtitle.lineBreakMode = .byWordWrapping

        configureDefaults()

        let tabView = NSTabView()
        tabView.translatesAutoresizingMaskIntoConstraints = false
        tabView.addTabViewItem(tab("Workflows", workflowsView()))
        tabView.addTabViewItem(tab("Search", searchView()))
        tabView.addTabViewItem(tab("Direct HF", directDownloadView()))
        tabView.addTabViewItem(tab("Audit & Prep", auditPrepView()))
        tabView.addTabViewItem(tab("Leaderboards", leaderboardsView()))

        let stack = vertical([title, subtitle, tabView])
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 22),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -22),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 22),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -22),
            subtitle.widthAnchor.constraint(equalTo: stack.widthAnchor),
            tabView.widthAnchor.constraint(equalTo: stack.widthAnchor),
            tabView.heightAnchor.constraint(greaterThanOrEqualToConstant: 610),
        ])
    }

    private func configureDefaults() {
        searchRootField.stringValue = defaultDownloadRoot()
        directRootField.stringValue = defaultDownloadRoot()
        conversionRootField.stringValue = defaultDownloadRoot()
        searchTermsField.placeholderString = "code OR cybersecurity, or code, coding"
        specificRepoField.placeholderString = "owner/model-name or huggingface.co URL"
        familyField.placeholderString = "Optional: qwen, qwen coder, deepseek"
        directRepoField.placeholderString = "owner/model-name"
        directFilterField.placeholderString = "Optional: Q4_K_M, model-q4_k_m.gguf, tokenizer.json"
        directExcludeField.placeholderString = "Optional: README.md, *.md, original/*"
        directEndpointField.placeholderString = "Optional custom Hugging Face endpoint"
        prepareOnlyField.placeholderString = "Optional: Ollama,GPT4All"
        prepareSkipField.placeholderString = "Optional: AIStudio,LocalAI"
        conversionSelectField.placeholderString = "Optional: 0/all, 3, or 2-5"

        specificKindPopup.addItems(withTitles: ["model", "dataset"])
        searchKindPopup.addItems(withTitles: ["models", "datasets", "both"])
        searchSourcePopup.addItems(withTitles: ["huggingface", "kaggle", "both"])
        directTypePopup.addItems(withTitles: ["model", "dataset"])
        verifyPopup.addItems(withTitles: ["size", "sha256", "none"])

        ggufCheck.state = .on
        coreMLCheck.state = .on
        hideDuplicatesCheck.state = .on
        auditDryRunCheck.state = .on
        prepareDryRunCheck.state = .on
        prepareContinueOnErrorCheck.state = .on
        conversionListOnlyCheck.state = .on
        conversionDryRunCheck.state = .on
        conversionNonInteractiveAuditCheck.state = .on
        codingBoardCheck.state = .on
        searchSpacesCheck.state = .on

        for button in [
            specificCheck, ggufCheck, coreMLCheck, mlxCheck, onnxCheck, safetensorsCheck,
            kerasCheck, pytorchCheck, anyArtifactCheck, hideDuplicatesCheck, dryRunCheck,
            auditDryRunCheck, prepareDryRunCheck, prepareNoAuditCheck, prepareNonInteractiveAuditCheck,
            prepareSkipDuplicatesAuditCheck, prepareContinueOnErrorCheck, conversionListOnlyCheck,
            conversionDryRunCheck, conversionNoAuditCheck, conversionNonInteractiveAuditCheck,
            conversionYesPrepareCheck, conversionForceCheck, generalBoardCheck, codingBoardCheck,
            visualBoardCheck, securityBoardCheck, embeddingBoardCheck, searchSpacesCheck,
            manualCacheCheck, refreshCacheCheck
        ] {
            button.setButtonType(.switch)
        }
    }

    private func workflowsView() -> NSView {
        let buttons = vertical([
            actionButton("Open Full modelmgr Menu", #selector(openFullMenu)),
            actionButton("Search/Download Default Flow", #selector(openSearchFlow)),
            actionButton("Local Audit", #selector(openAuditFlow)),
            actionButton("Leaderboards", #selector(openLeaderboardsFlow)),
            actionButton("Relaunch Native GUI", #selector(openGuiFlow)),
        ])
        buttons.alignment = .leading

        let note = wrappedLabel("""
        These buttons open the exact model_manager.py flows in Terminal. Use the Search tab when you want the GUI to pre-fill the search source, model types, size range, exclusions, family filters, and result limits before the interactive result picker starts.
        """)
        return padded(vertical([section("Top-level modelmgr options"), note, buttons, separator(), toolsRow()]))
    }

    private func searchView() -> NSView {
        let artifactRow = horizontal([
            ggufCheck, coreMLCheck, mlxCheck, onnxCheck,
            safetensorsCheck, kerasCheck, pytorchCheck, anyArtifactCheck
        ])

        let rootBrowse = NSButton(title: "Browse", target: self, action: #selector(browseSearchRoot))
        let grid = formGrid([
            ("Specific repo", horizontal([specificCheck])),
            ("Repo ID / URL", horizontal([specificRepoField, specificKindPopup])),
            ("Search terms", searchTermsField),
            ("Search for", searchKindPopup),
            ("Search source", searchSourcePopup),
            ("Artifact types", artifactRow),
            ("Model size range", sizeRangeField),
            ("Authors to exclude", authorsField),
            ("Terms/tags to exclude", termsField),
            ("Families to exclude", familyField),
            ("Results per term", resultLimitField),
            ("Display batch size", pageSizeField),
            ("Duplicate handling", hideDuplicatesCheck),
            ("Download root", horizontal([searchRootField, rootBrowse])),
        ])

        let run = actionButton("Run Guided Search/Download in Terminal", #selector(runGuidedSearch))
        let note = wrappedLabel("The GUI sends these options to model_manager.py, then Terminal stays interactive for result selection, artifact picking, scanning, delete prompts, prep, and conversion.")
        return padded(vertical([section("Guided search/download"), note, grid, horizontal([run])]))
    }

    private func directDownloadView() -> NSView {
        let browseButton = NSButton(title: "Browse", target: self, action: #selector(browseDirectRoot))
        startButton.target = self
        startButton.action = #selector(startDirectDownload)
        stopButton.target = self
        stopButton.action = #selector(stopDownload)
        stopButton.isEnabled = false

        outputView.isEditable = false
        outputView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        let scroll = NSScrollView()
        scroll.borderType = .bezelBorder
        scroll.hasVerticalScroller = true
        scroll.documentView = outputView

        let grid = formGrid([
            ("Repo ID", directRepoField),
            ("Type", directTypePopup),
            ("Download root", horizontal([directRootField, browseButton])),
            ("Filters", directFilterField),
            ("Excludes", directExcludeField),
            ("Endpoint", directEndpointField),
            ("Connections", connectionsField),
            ("Concurrent files", maxActiveField),
            ("Verify", verifyPopup),
            ("Mode", dryRunCheck),
        ])
        let note = wrappedLabel("Direct download uses hfdownloader in this window. It is useful when you already know the exact Hugging Face repo and filter terms.")
        let stack = vertical([section("Direct Hugging Face download"), note, grid, horizontal([startButton, stopButton]), scroll])
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 250).isActive = true
        return padded(stack)
    }

    private func auditPrepView() -> NSView {
        let audit = actionButton("Run Local Audit", #selector(runAuditWithOptions))
        let prepare = actionButton("Run Prepare_models_for_All.py", #selector(runPrepareAll))
        let conversion = actionButton("Run model_conversion.py", #selector(runConversionHelper))
        let conversionBrowse = NSButton(title: "Browse", target: self, action: #selector(browseConversionRoot))
        let install = actionButton("Install / Rebuild hfdownloader", #selector(runInstallDownloader))
        let openRoot = actionButton("Open Download Root", #selector(openDownloadRoot))
        let openTools = actionButton("Open Tools Folder", #selector(openToolsFolder))

        let prepareGrid = formGrid([
            ("Workers", prepareWorkersField),
            ("Only apps", prepareOnlyField),
            ("Skip apps", prepareSkipField),
            ("Audit flags", horizontal([prepareNoAuditCheck, prepareNonInteractiveAuditCheck, prepareSkipDuplicatesAuditCheck])),
            ("Run mode", horizontal([prepareDryRunCheck, prepareContinueOnErrorCheck])),
        ])
        let conversionGrid = formGrid([
            ("Root", horizontal([conversionRootField, conversionBrowse])),
            ("Quant", conversionQuantField),
            ("Workers", conversionWorkersField),
            ("Select", conversionSelectField),
            ("Run mode", horizontal([conversionListOnlyCheck, conversionDryRunCheck, conversionForceCheck])),
            ("Audit/prepare", horizontal([conversionNoAuditCheck, conversionNonInteractiveAuditCheck, conversionYesPrepareCheck])),
        ])

        let note = wrappedLabel("Audit, prep, conversion, and installer commands run in Terminal so their prompts, warnings, and destructive confirmations stay visible.")
        return padded(vertical([
            section("Audit"),
            note,
            horizontal([auditDryRunCheck, audit]),
            separator(),
            section("Prepare all apps"),
            prepareGrid,
            horizontal([prepare]),
            separator(),
            section("Safetensors to GGUF conversion"),
            conversionGrid,
            horizontal([conversion]),
            separator(),
            section("Tools"),
            horizontal([install, openRoot, openTools]),
        ]))
    }

    private func leaderboardsView() -> NSView {
        let categories = horizontal([generalBoardCheck, codingBoardCheck, visualBoardCheck, securityBoardCheck, embeddingBoardCheck])
        let run = actionButton("Show Leaderboards", #selector(runLeaderboardsWithOptions))
        let grid = formGrid([
            ("Categories", categories),
            ("HF Spaces", searchSpacesCheck),
            ("Manual cache", manualCacheCheck),
            ("Cache refresh", refreshCacheCheck),
            ("Cache limit", leaderboardLimitField),
        ])
        let note = wrappedLabel("Leaderboard sources are printed in Terminal. Optional cache refresh uses Hugging Face searches to seed local leaderboard-aware recommendations.")
        return padded(vertical([section("Leaderboards"), note, grid, horizontal([run])]))
    }

    private func toolsRow() -> NSView {
        horizontal([
            actionButton("Install hfdownloader", #selector(runInstallDownloader)),
            actionButton("Open Download Root", #selector(openDownloadRoot)),
            actionButton("Open Tools Folder", #selector(openToolsFolder)),
        ])
    }

    private func tab(_ title: String, _ view: NSView) -> NSTabViewItem {
        let item = NSTabViewItem()
        item.label = title
        item.view = view
        return item
    }

    private func padded(_ view: NSView) -> NSView {
        let container = NSView()
        container.translatesAutoresizingMaskIntoConstraints = false
        view.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(view)
        NSLayoutConstraint.activate([
            view.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 14),
            view.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -14),
            view.topAnchor.constraint(equalTo: container.topAnchor, constant: 14),
            view.bottomAnchor.constraint(lessThanOrEqualTo: container.bottomAnchor, constant: -14),
        ])
        return container
    }

    private func section(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = NSFont.boldSystemFont(ofSize: 14)
        field.textColor = .secondaryLabelColor
        return field
    }

    private func wrappedLabel(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.textColor = .secondaryLabelColor
        field.lineBreakMode = .byWordWrapping
        field.maximumNumberOfLines = 0
        return field
    }

    private func label(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.textColor = .secondaryLabelColor
        return field
    }

    private func separator() -> NSBox {
        let box = NSBox()
        box.boxType = .separator
        return box
    }

    private func actionButton(_ title: String, _ action: Selector) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.bezelStyle = .rounded
        return button
    }

    private func vertical(_ views: [NSView]) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        return stack
    }

    private func horizontal(_ views: [NSView]) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 8
        return stack
    }

    private func formGrid(_ rows: [(String, NSView)]) -> NSGridView {
        let grid = NSGridView(views: rows.map { [label($0.0), $0.1] })
        grid.column(at: 0).xPlacement = .trailing
        grid.column(at: 1).xPlacement = .fill
        grid.rowSpacing = 10
        grid.columnSpacing = 12
        grid.translatesAutoresizingMaskIntoConstraints = false
        for (_, view) in rows {
            if let field = view as? NSTextField {
                field.widthAnchor.constraint(greaterThanOrEqualToConstant: 560).isActive = true
            }
        }
        return grid
    }

    private func defaultDownloadRoot() -> String {
        let args = CommandLine.arguments
        if let idx = args.firstIndex(of: "--download-root"), idx + 1 < args.count {
            return args[idx + 1]
        }
        if let env = ProcessInfo.processInfo.environment["MODEL_MANAGER_DOWNLOAD_DIR"], !env.isEmpty {
            return env
        }
        return NSHomeDirectory() + "/models"
    }

    private func scriptDirectory() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    }

    private func modelManagerScriptPath() -> String {
        scriptDirectory().appendingPathComponent("model_manager.py").path
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private func appleScriptString(_ value: String) -> String {
        "\"" + value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n") + "\""
    }

    private func openTerminal(command: String) {
        let source = """
        tell application "Terminal"
            activate
            do script \(appleScriptString(command))
        end tell
        """

        var error: NSDictionary?
        if let appleScript = NSAppleScript(source: source) {
            appleScript.executeAndReturnError(&error)
        }
        if let error {
            append("ERROR: could not open Terminal: \(error)\n")
        } else {
            append("Opened Terminal: \(command)\n")
        }
    }

    private func openManagerInTerminal(arguments: [String]) {
        let dir = scriptDirectory().path
        let script = modelManagerScriptPath()
        let args = arguments.map(shellQuote).joined(separator: " ")
        let command = "cd \(shellQuote(dir)) && python3 \(shellQuote(script)) \(args)"
        openTerminal(command: command)
    }

    private func openScriptInTerminal(_ scriptName: String, arguments: [String] = []) {
        let dir = scriptDirectory().path
        let script = scriptDirectory().appendingPathComponent(scriptName).path
        let args = arguments.map(shellQuote).joined(separator: " ")
        let command = "cd \(shellQuote(dir)) && python3 \(shellQuote(script)) \(args)"
        openTerminal(command: command)
    }

    private func openShellScriptInTerminal(_ scriptName: String) {
        let dir = scriptDirectory().path
        let script = scriptDirectory().appendingPathComponent(scriptName).path
        let command = "cd \(shellQuote(dir)) && \(shellQuote(script))"
        openTerminal(command: command)
    }

    private func downloaderPath() -> String? {
        let env = ProcessInfo.processInfo.environment["MODEL_MANAGER_HFDOWNLOADER_BIN"]
        if let env, !env.isEmpty, FileManager.default.isExecutableFile(atPath: env) {
            return env
        }
        let local = scriptDirectory().appendingPathComponent("bin/hfdownloader").path
        if FileManager.default.isExecutableFile(atPath: local) {
            return local
        }
        return findOnPath("hfdownloader")
    }

    private func findOnPath(_ name: String) -> String? {
        let pathEnv = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for dir in pathEnv.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(dir)).appendingPathComponent(name).path
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    private func selectedArtifactTypes() -> String {
        if anyArtifactCheck.state == .on {
            return "any"
        }
        var values: [String] = []
        if ggufCheck.state == .on { values.append("gguf") }
        if coreMLCheck.state == .on { values.append("coreml") }
        if mlxCheck.state == .on { values.append("mlx") }
        if onnxCheck.state == .on { values.append("onnx") }
        if safetensorsCheck.state == .on { values.append("safetensors") }
        if kerasCheck.state == .on { values.append("keras") }
        if pytorchCheck.state == .on { values.append("pytorch") }
        return values.isEmpty ? "gguf,coreml" : values.joined(separator: ",")
    }

    private func selectedLeaderboardCategories() -> String {
        var values: [String] = []
        if generalBoardCheck.state == .on { values.append("general") }
        if codingBoardCheck.state == .on { values.append("coding") }
        if visualBoardCheck.state == .on { values.append("visual") }
        if securityBoardCheck.state == .on { values.append("security") }
        if embeddingBoardCheck.state == .on { values.append("embedding") }
        return values.isEmpty ? "coding" : values.joined(separator: ",")
    }

    @objc private func openFullMenu() {
        openManagerInTerminal(arguments: [])
    }

    @objc private func openSearchFlow() {
        openManagerInTerminal(arguments: ["--search"])
    }

    @objc private func openAuditFlow() {
        openManagerInTerminal(arguments: ["--audit"])
    }

    @objc private func openLeaderboardsFlow() {
        openManagerInTerminal(arguments: ["--leaderboards"])
    }

    @objc private func openGuiFlow() {
        openManagerInTerminal(arguments: ["--gui"])
    }

    @objc private func runGuidedSearch() {
        var args = ["--search", "--download-root", searchRootField.stringValue]
        if specificCheck.state == .on {
            let repo = specificRepoField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard repo.contains("/") else {
                append("ERROR: enter a Hugging Face repo ID or URL for the specific repo search.\n")
                return
            }
            args.append(contentsOf: ["--specific-repo", repo, "--specific-kind", specificKindPopup.titleOfSelectedItem ?? "model"])
        } else {
            let query = searchTermsField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !query.isEmpty else {
                append("ERROR: enter search terms or enable specific repo mode.\n")
                return
            }
            args.append(contentsOf: [
                "--search-query", query,
                "--search-kind", searchKindPopup.titleOfSelectedItem ?? "models",
                "--search-source", searchSourcePopup.titleOfSelectedItem ?? "huggingface",
                "--artifact-types", selectedArtifactTypes(),
                "--model-size-range", sizeRangeField.stringValue,
                "--exclude-publishers", authorsField.stringValue,
                "--exclude-terms", termsField.stringValue,
                "--result-limit", resultLimitField.stringValue,
                "--page-size", pageSizeField.stringValue,
            ])
            let families = familyField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            args.append(contentsOf: ["--exclude-families", families.isEmpty ? "none" : families])
            args.append(hideDuplicatesCheck.state == .on ? "--hide-duplicate-families" : "--show-duplicate-families")
        }
        openManagerInTerminal(arguments: args)
    }

    @objc private func runAuditWithOptions() {
        openManagerInTerminal(arguments: ["--audit", auditDryRunCheck.state == .on ? "--audit-dry-run" : "--audit-live"])
    }

    @objc private func runPrepareAll() {
        var args: [String] = []
        let workers = prepareWorkersField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !workers.isEmpty {
            args.append(contentsOf: ["--workers", workers])
        }
        if prepareDryRunCheck.state == .on {
            args.append("--dry-run")
        }
        if prepareNoAuditCheck.state == .on {
            args.append("--no-audit")
        }
        if prepareNonInteractiveAuditCheck.state == .on {
            args.append("--non-interactive-audit")
        }
        if prepareSkipDuplicatesAuditCheck.state == .on {
            args.append("--skip-duplicates-audit")
        }
        let only = prepareOnlyField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !only.isEmpty {
            args.append(contentsOf: ["--only", only])
        }
        let skip = prepareSkipField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !skip.isEmpty {
            args.append(contentsOf: ["--skip", skip])
        }
        if prepareContinueOnErrorCheck.state == .on {
            args.append("--continue-on-error")
        }
        openScriptInTerminal("Prepare_models_for_All.py", arguments: args)
    }

    @objc private func runConversionHelper() {
        var args: [String] = []
        if conversionListOnlyCheck.state == .on {
            args.append("--list-only")
        }
        if conversionDryRunCheck.state == .on {
            args.append("--dry-run")
        }
        if conversionNoAuditCheck.state == .on {
            args.append("--no-audit")
        }
        if conversionNonInteractiveAuditCheck.state == .on {
            args.append("--non-interactive-audit")
        }
        let quant = conversionQuantField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !quant.isEmpty {
            args.append(contentsOf: ["--quant", quant])
        }
        let workers = conversionWorkersField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !workers.isEmpty {
            args.append(contentsOf: ["--workers", workers])
        }
        let root = conversionRootField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !root.isEmpty {
            args.append(contentsOf: ["--root", root])
        }
        let selection = conversionSelectField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selection.isEmpty {
            args.append(contentsOf: ["--select", selection])
        }
        if conversionYesPrepareCheck.state == .on {
            args.append("--yes-prepare")
        }
        if conversionForceCheck.state == .on {
            args.append("--force")
        }
        openScriptInTerminal("model_conversion.py", arguments: args)
    }

    @objc private func runInstallDownloader() {
        openShellScriptInTerminal("install_hfdownloader.sh")
    }

    @objc private func runLeaderboardsWithOptions() {
        var args = ["--leaderboards", "--leaderboard-categories", selectedLeaderboardCategories()]
        if searchSpacesCheck.state == .off {
            args.append("--skip-hf-spaces")
        }
        if manualCacheCheck.state == .on {
            args.append("--manual-leaderboard-cache")
        }
        if refreshCacheCheck.state == .on {
            args.append("--refresh-leaderboard-cache")
            args.append(contentsOf: ["--leaderboard-cache-limit", leaderboardLimitField.stringValue])
        }
        openManagerInTerminal(arguments: args)
    }

    @objc private func browseSearchRoot() {
        browseDirectory(into: searchRootField)
    }

    @objc private func browseDirectRoot() {
        browseDirectory(into: directRootField)
    }

    @objc private func browseConversionRoot() {
        browseDirectory(into: conversionRootField)
    }

    private func browseDirectory(into field: NSTextField) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            field.stringValue = url.path
        }
    }

    @objc private func openDownloadRoot() {
        let path = searchRootField.stringValue.isEmpty ? defaultDownloadRoot() : searchRootField.stringValue
        NSWorkspace.shared.open(URL(fileURLWithPath: path, isDirectory: true))
    }

    @objc private func openToolsFolder() {
        NSWorkspace.shared.open(scriptDirectory())
    }

    @objc private func startDirectDownload() {
        guard process == nil else { return }
        guard let downloader = downloaderPath() else {
            append("ERROR: hfdownloader not found. Run install_hfdownloader.sh first.\n")
            return
        }

        let repo = directRepoField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard repo.contains("/") else {
            append("ERROR: enter a Hugging Face repo ID like owner/name.\n")
            return
        }

        let type = directTypePopup.titleOfSelectedItem ?? "model"
        let root = directRootField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = URL(fileURLWithPath: root)
            .appendingPathComponent("huggingface")
            .appendingPathComponent(type)
            .path
        try? FileManager.default.createDirectory(atPath: base, withIntermediateDirectories: true)

        var args = ["download", repo, "--local-dir", base]
        if type == "dataset" {
            args.append("--dataset")
        }
        let filters = directFilterField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !filters.isEmpty {
            args.append(contentsOf: ["--filters", filters])
        }
        let excludes = directExcludeField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !excludes.isEmpty {
            args.append(contentsOf: ["--exclude", excludes])
        }
        let endpoint = directEndpointField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !endpoint.isEmpty {
            args.append(contentsOf: ["--endpoint", endpoint])
        }
        args.append(contentsOf: ["--connections", connectionsField.stringValue])
        args.append(contentsOf: ["--max-active", maxActiveField.stringValue])
        args.append(contentsOf: ["--verify", verifyPopup.titleOfSelectedItem ?? "size"])
        if dryRunCheck.state == .on {
            args.append(contentsOf: ["--dry-run", "--plan-format", "json"])
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: downloader)
        proc.arguments = args

        var env = ProcessInfo.processInfo.environment
        if env["HF_TOKEN"] == nil, let token = env["HUGGINGFACEHUB_API_TOKEN"] {
            env["HF_TOKEN"] = token
        }
        proc.environment = env

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self?.append(text)
            }
        }

        proc.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async {
                self?.append("\nProcess exited with code \(finished.terminationStatus).\n")
                self?.process = nil
                self?.startButton.isEnabled = true
                self?.stopButton.isEnabled = false
            }
        }

        append("\nRunning: \(downloader) \(args.joined(separator: " "))\n")
        do {
            process = proc
            startButton.isEnabled = false
            stopButton.isEnabled = true
            try proc.run()
        } catch {
            process = nil
            startButton.isEnabled = true
            stopButton.isEnabled = false
            append("ERROR: \(error.localizedDescription)\n")
        }
    }

    @objc private func stopDownload() {
        process?.terminate()
    }

    private func append(_ text: String) {
        outputView.textStorage?.append(NSAttributedString(string: text))
        outputView.scrollToEndOfDocument(nil)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
