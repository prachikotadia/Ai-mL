const fs = require('fs');
const path = require('path');

const inputDir = path.join(__dirname, '..');
const outputFile = path.join(__dirname, 'src', 'data', 'cards.json');

try {
  const files = fs.readdirSync(inputDir).filter(file => file.startsWith('flashcards_section_') && file.endsWith('.md'));
  const allCards = [];

  files.sort((a, b) => {
    const numA = parseInt(a.match(/section_(\d+)/)[1]);
    const numB = parseInt(b.match(/section_(\d+)/)[1]);
    return numA - numB;
  });

  files.forEach(file => {
    const data = fs.readFileSync(path.join(inputDir, file), 'utf8');

    // Extract Section Name from the first line like "# Section 1: Machine Learning Basics"
    const sectionTitleMatch = data.match(/^# (Section \d+: .*)$/m);
    const sectionTitle = sectionTitleMatch ? sectionTitleMatch[1] : "General";
    // Extract a simple ID like "section-1" from the filename "flashcards_section_1.md"
    const sectionIdMatch = file.match(/section_(\d+)/);
    const sectionId = sectionIdMatch ? `section-${sectionIdMatch[1]}` : `section-${file}`;

    const sections = data.split(/^##\s+/m);
    const sectionCards = [];

    sections.forEach((section, index) => {
      // First split element might be the file header, so skip if no ID found
      const titleMatch = section.match(/^(\d+)\.\s+(.*)$/m);
      if (!titleMatch) return;

      const id = parseInt(titleMatch[1]);
      const lines = section.split('\n');
      const headerLine = lines[0].trim();
      const title = headerLine.replace(/^\d+\.\s+/, '');

      // Extract parts
      const questionMatch = section.match(/🟦\s+\*\*(.*?)\*\*/);
      const question = questionMatch ? questionMatch[1] : '';

      const defMatch = section.match(/🟩\s+\*\*Definition\*\*\n([\s\S]*?)(\n🟨|$)/);
      const definition = defMatch ? defMatch[1].trim() : '';

      const exMatch = section.match(/🟨\s+\*\*How It Works \/ Example\*\*\n([\s\S]*?)(\n🟪|$)/);
      const example = exMatch ? exMatch[1].trim() : '';

      const tipMatch = section.match(/🟪\s+\*\*Quick Tip\*\*\n([\s\S]*?)(\n---|$)/);
      const tip = tipMatch ? tipMatch[1].trim() : '';

      if (question && definition) {
        sectionCards.push({
          id,
          title,
          question,
          definition,
          example,
          tip
        });
      }
    });

    if (sectionCards.length > 0) {
      allCards.push({
        id: sectionId,
        title: sectionTitle,
        itemCount: sectionCards.length,
        cards: sectionCards
      });
    }
  });

  fs.writeFileSync(outputFile, JSON.stringify(allCards, null, 2));
  console.log(`Successfully parsed ${allCards.length} sections to ${outputFile}`);

} catch (err) {
  process.exit(1);
}
