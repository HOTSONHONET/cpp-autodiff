#pragma once
#include "layers.hpp"

struct VIT {
    PatchEmbedding patch_embed;
    vector<TransformerBlock> blocks;
    Linear head;

    VIT(
        int img_size,
        int patch_size,
        int embed_dim,
        int depth,
        int num_heads,
        int num_classes
    ) : patch_embed(img_size, patch_size, embed_dim), head(embed_dim, num_classes) {

        for(int i = 0; i < depth; i++) {
            blocks.emplace_back(embed_dim, num_heads);
        }
    }

    vector<Value*> forward(vector<vector<Value*>> image) {
        auto tokens = patch_embed.forward(image);

        for(auto &block: blocks){
            tokens = block.forward(tokens);
        }

        return head(tokens[0]); // CLS token
    }
}