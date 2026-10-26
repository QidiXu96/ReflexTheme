# ReflexTheme
## **Model Summary** 
We developed Reflexive Thematic Agents (ReflexTheme) to support RTA with LLMs. ReflexTheme integrates several specialized agents. First, multiple coder agents with different identity profiles and reflections collaboratively analyze each transcript to identify excerpts and generate initial codes. The reviewer agent merges similar codes across transcripts to construct a codebook. Based on this codebook, multiple themer agents generate candidate themes, while the repairer agent ensures that all identified codes are appropriately represented in the themes.
<img width="3004" height="2452" alt="new reflextheme (2)" src="https://github.com/user-attachments/assets/11306a5b-2a53-4c48-b987-d3b5b86aa0f3" />

## **API Requirement** 
Users must provide their own API credentials to run ReflexTheme. 

## **Usage**
1. Clone or download this repository. Related ReflexTheme coding functions at [ReflexTheme_Utils](./ReflexTheme_Utils/). <br>
2. Prepare your own API credentials.
3. Open [ReflexTheme_Covid.ipynb](./ReflexTheme_Covid.ipynb/).
4. Follow the notebook to configure the API, load transcripts, and run ReflexTheme.
