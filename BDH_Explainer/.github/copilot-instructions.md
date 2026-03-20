<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	- Project: Interactive BDH (Dragon Hatchling) Visualizer
	- Tech Stack: Svelte (frontend), D3.js (visualizations), Python FastAPI (backend), KaTeX (math rendering)
	- Features: Hebbian learning, Linear attention, FFN with sparse activations, Scale-free networks, Monosemanticity, Weight matrix visualization

- [x] Scaffold the Project
	- Created frontend structure with Svelte SPA
	- Created backend structure with Flask API
	- Set up configuration files (vite.config.ts, tsconfig.json, package.json)

- [x] Customize the Project
	- Implemented 7 visualization components for BDH concepts
	- Added interactive features for step-by-step token processing
	- Created D3.js visualizations for matrices, graphs, and networks

- [x] Install Required Extensions
	- Installed Svelte for VS Code extension

- [x] Compile the Project
	- Installed npm dependencies (120 packages)
	- Backend dependencies specified in requirements.txt
	- Build process tested successfully

- [x] Create and Run Task
	- Created task "Run BDH Visualizer (Frontend)"
	- Frontend dev server running on http://localhost:3000

- [x] Launch the Project
	- Development server is running at http://localhost:3000
	- Backend can be started with: cd backend && python app.py

- [x] Ensure Documentation is Complete
	- Created comprehensive README.md with usage instructions
	- Documented all 7 visualization features
	- Added quick start guide and project structure
