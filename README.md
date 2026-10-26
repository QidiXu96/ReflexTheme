# ReflexTheme
We developed Reflexive Thematic Agents (ReflexTheme) to support RTA with LLMs. ReflexTheme integrates several specialized agents. First, multiple coder agents with different identity profiles and reflections collaboratively analyze each transcript to identify excerpts and generate initial codes. The reviewer agent merges similar codes across transcripts to construct a codebook. Based on this codebook, multiple themer agents generate candidate themes, while the repairer agent ensures that all identified codes are appropriately represented in the themes.
<img width="3004" height="2452" alt="new reflextheme (1)" src="https://github.com/user-attachments/assets/4504deca-09f3-4875-ac11-d8096cd88ee4" />

Related ReflexTheme coding functions at [ReflexTheme_Utils](./ReflexTheme_Utils/). <br>
Apply ReflexTheme into covid example transcripts at [ReflexTheme_Covid.ipynb](./ReflexTheme_Covid.ipynb/).
