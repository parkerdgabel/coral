# Coral Data Format Specification

## Table of Contents

1. [Introduction](#1-introduction)
2. [Key Concepts](#2-key-concepts)
    - [2.1 Immutable Data Model](#21-immutable-data-model)
    - [2.2 Hierarchical Organization in Deltas](#22-hierarchical-organization-in-deltas)
    - [2.3 Tiling and Chunking](#23-tiling-and-chunking)
    - [2.4 Version Control and Deltas](#24-version-control-and-deltas)
    - [2.5 Transactions and Manifests](#25-transactions-and-manifests)
    - [2.6 Immutable Branches and Tags](#26-immutable-branches-and-tags)
3. [Store Specification](#3-store-specification)
    - [3.1 Required Store Properties](#31-required-store-properties)
    - [3.2 Atomic Operations](#32-atomic-operations)
    - [3.3 Consistency Models](#33-consistency-models)
    - [3.4 Namespace and Naming Constraints](#34-namespace-and-naming-constraints)
4. [File Structure](#4-file-structure)
    - [4.1 Directory Layout](#41-directory-layout)
    - [4.2 Tile Files and Naming Conventions](#42-tile-files-and-naming-conventions)
    - [4.3 Manifests and Metadata Handling](#43-manifests-and-metadata-handling)
    - [4.4 Branch and Tag References](#44-branch-and-tag-references)
5. [Data Model and Schemas](#5-data-model-and-schemas)
    - [5.1 Enumerations and Unions](#51-enumerations-and-unions)
    - [5.2 Tables](#52-tables)
    - [5.3 FlatBuffers Schemas](#53-flatbuffers-schemas)
6. [Data Format Logic and Workflow](#6-data-format-logic-and-workflow)
    - [6.1 Tiling Algorithm](#61-tiling-algorithm)
    - [6.2 Composable Codecs](#62-composable-codecs)
    - [6.3 Delta Commit Process](#63-delta-commit-process)
    - [6.4 Dataset Reconstruction](#64-dataset-reconstruction)
7. [Atomicity and Consistency](#7-atomicity-and-consistency)
    - [7.1 Atomicity](#71-atomicity)
    - [7.2 Consistency](#72-consistency)
8. [Examples](#8-examples)
    - [8.1 Array Creation and Reshaping](#81-array-creation-and-reshaping)
    - [8.2 Tile Storage and Retrieval](#82-tile-storage-and-retrieval)
    - [8.3 Transaction and Manifest Creation](#83-transaction-and-manifest-creation)
    - [8.4 Commit and Conflict Resolution](#84-commit-and-conflict-resolution)
9. [Implementation Considerations](#9-implementation-considerations)
    - [9.1 Storage Backend Compatibility](#91-storage-backend-compatibility)
    - [9.2 Data Integrity Verification](#92-data-integrity-verification)
10. [Conclusion](#10-conclusion)

---

### 1. Introduction

Coral is a high-performance, immutable data format designed for efficient management and storage of large-scale tensor data. Optimized for use with various storage backends, including local file systems and object stores like Amazon S3, Coral supports:

- **Immutable Data Model**: Ensures data integrity and simplifies concurrency control by making all data and metadata immutable once written.
- **Chunked Tiling**: Partitions arrays into tiles (chunks) according to customizable chunk grids, enhancing storage efficiency and data access performance.
- **Composable Codecs**: Utilizes flexible data compression and transformation pipelines through codecs, allowing for optimized storage and interoperability.
- **Version Control**: Implements a robust versioning system with support for transactions, deltas, manifests, branches, and tags.
- **Atomicity and Consistency**: Maintains data integrity through atomic transactions and ensures consistent dataset states.
- **Immutable Branches and Tags**: Branch references are stored as immutable files, enhancing consistency and enabling atomic updates.

This specification provides a comprehensive reference for the Coral data format, detailing file structures, data models, schemas, workflows, and store requirements necessary for implementation and interoperability.

---

### 2. Key Concepts

#### 2.1 Immutable Data Model

- **Immutability**: Once written, data and metadata files are never modified or deleted. Changes are made by creating new files representing new versions.
- **Versioned Updates**: All updates, including to array and group metadata, are recorded as new immutable deltas.
- **Benefits**:
  - **Data Integrity**: Prevents corruption due to concurrent writes.
  - **Auditability**: Provides a complete history of changes.
  - **Consistency**: Simplifies reasoning about dataset states.

#### 2.2 Hierarchical Organization in Deltas

- **Groups and Arrays in Deltas**: All group and array metadata is stored within `UpdateGroup` and `UpdateArray` deltas.
- **Hierarchy Initialization**: The initial deltas in the dataset are typically `UpdateGroup` or `UpdateArray` deltas to establish the hierarchy.
- **Manifests Reference Metadata**: Manifests store references to the most recent array and group metadata for quick retrieval.

#### 2.3 Tiling and Chunking

- **Tiles (Chunks)**: Arrays are partitioned into tiles according to a chunk grid, facilitating efficient storage and access.
- **Chunk Grid**: Defined by the `chunks` parameter in array metadata, specifying the size of chunks along each dimension.
- **Tile Identification**:
  - Tiles are identified by their array name, grid indices, and delta IDs.
  - File names include these identifiers for easy retrieval.

#### 2.4 Version Control and Deltas

- **Deltas**: Represent changes to the dataset, such as adding, modifying, or deleting tiles, and updates to array and group metadata.
- **Delta Types**:
  - `AddTile`: Adding a new tile.
  - `ModifyTile`: Modifying an existing tile.
  - `DeleteTile`: Deleting a tile.
  - `UpdateArray`: Updating an array’s metadata or structure.
  - `UpdateGroup`: Updating a group’s metadata or structure.
  - `UpdateAttributes`: Updating attributes.
- **Immutable Deltas**: Once a delta is created, it is never modified or deleted.

#### 2.5 Transactions and Manifests

- **Transactions**: Group multiple deltas into atomic units of work.
- **Manifests**:
  - Represent specific dataset versions, aggregating deltas.
  - Store references to the most recent array and group metadata for quick retrieval.
  - **Immutability**: Transactions and manifests are immutable once written.

#### 2.6 Immutable Branches and Tags

- **Immutable Branch References**: Branch references are stored as immutable files using a sequence-numbered naming convention.
- **Branch Updates**:
  - New branch references are created with incremented sequence numbers.
  - The latest branch reference is determined by lexicographical sorting of the sequence numbers.
- **Atomic Branch Updates**:
  - Updates to branches are performed using atomic "create if not exists" operations.
  - Ensures consistency even in storage systems with limited atomicity guarantees.
- **Tags**:
  - Immutable references to specific manifests.
  - Once created, tags cannot be modified or deleted.

---

### 3. Store Specification

This section defines the required properties and behaviors of the storage system (the "store") necessary to ensure atomicity and consistency in the Coral data format.

#### 3.1 Required Store Properties

Implementers must ensure that the chosen storage backend provides the following properties:

1. **Atomic File Creation**:
    - The ability to perform atomic "create if not exists" operations for files.
    - Guarantees that a file creation operation will fail if the file already exists, preventing race conditions.
2. **Consistent Namespace**:
    - A global namespace where files are uniquely identified by their paths.
    - Ensures that file paths are consistent and collision-free across concurrent operations.
3. **Eventual Consistency Handling**:
    - If the store provides eventual consistency (e.g., S3), implementers must handle possible stale reads and ensure data integrity through retries and conflict detection.
4. **Lexicographical Listing**:
    - Ability to list files or objects in lexicographical order based on their names.
    - Essential for determining the latest branch reference by sorting sequence-numbered files.
5. **Read-after-Write Consistency for New Objects**:
    - The store should provide read-after-write consistency for new files to ensure that once a file is written, it is immediately available for reading.

#### 3.2 Atomic Operations

To maintain atomicity and consistency, the store must support:

- **Atomic "Create If Not Exists" Operation**:
  - The operation must ensure that if a file does not exist, it is created; if it already exists, the operation fails without modifying the existing file.
  - Used for creating branch reference files during commits to prevent conflicting updates.
- **Atomic File Writes**:
  - Writing a file must be an all-or-nothing operation; partial writes must not occur.
  - Ensures that files (e.g., tiles, manifests, transactions) are fully written before they become visible to readers.

#### 3.3 Consistency Models

Implementers must understand and handle the consistency model of the storage backend:

- **Strong Consistency**:
  - Changes are immediately visible to all clients.
  - Simplifies implementation but is not always available in object stores.
- **Eventual Consistency**:
  - Changes may not be immediately visible to other clients.
  - Implementers must handle potential stale reads and ensure data integrity through:
     - **Retries**: Retry read operations if inconsistencies are detected.
     - **Conflict Detection**: Use atomic operations to detect conflicting updates.

#### 3.4 Namespace and Naming Constraints

- **Unique File Paths**:
  - The store must support unique identification of files based on their paths.
  - File names must be case-sensitive if the store supports case-sensitive file systems.
- **Character Restrictions**:
  - File and directory names must not contain characters disallowed by the storage backend.
  - Branch and tag names must not contain the `/` character.

---

### 4. File Structure

#### 4.1 Directory Layout

The Coral data format organizes files in a flat directory structure, focusing on tiles, transactions, manifests, and immutable branch references.

```
/
├── tiles/
│   ├── tile_{array_name}_{grid_indices}_{delta_id}
│   ├── ...
├── transactions/
│   └── tx-{transaction_id}.fbs
├── manifests/
│   └── manifest_{manifest_id}.fbs
├── refs/
│   ├── branch.{branch_name}/
│   │   └── {encoded_sequence}.json
│   └── tag.{tag_name}/
│       └── ref.json
```

**Notes**:

- `tiles/`: Contains all tile files. Each tile file name encodes the array name, grid indices, and delta ID.
- `transactions/`: Contains serialized transaction files, each representing a set of deltas committed together.
- `manifests/`: Contains manifest files representing dataset versions.
- `refs/`:
  - `branch.{branch_name}/`: Contains immutable branch references with sequence-numbered file names.
  - `tag.{tag_name}/`: Contains immutable tag references.

#### 4.2 Tile Files and Naming Conventions

**Tile File Naming**:

```
tiles/tile_{array_name}_{grid_indices}_{delta_id}
```

- `{array_name}`: Name of the array.
- `{grid_indices}`: Grid indices, joined by underscores (e.g., `0_1_2`).
- `{delta_id}`: Identifier of the delta that introduced or modified the tile.

**Example**:

```
tiles/tile_temperature_grid_0_0_delta_0001
tiles/tile_temperature_grid_0_1_delta_0001
```

#### 4.3 Manifests and Metadata Handling

- **Manifests**:
  - Include references to the most recent array and group metadata IDs.
  - Embed the hierarchy of groups and arrays directly.
  - Stored as immutable files in `manifests/`.
- **Hierarchy in Deltas**:
  - All group and array metadata is stored within `UpdateGroup` and `UpdateArray` deltas.
  - Deltas are included in transactions and referenced in manifests.

#### 4.4 Branch and Tag References

**Branch References**:

- **Directory Structure**:

  ```
  refs/
  └── branch.{branch_name}/
        └── {encoded_sequence}.json
  ```

  - `{branch_name}`: Name of the branch (no `/` character).
  - `{encoded_sequence}`: Encoded sequence number.

- **Sequence Number Encoding**:
  - **Total Possible Commits**: Up to 2<sup>40</sup> - 1 (1,099,511,627,775).
  - **Sequence Number N**: Incremented for each commit.
  - **Encoded Sequence**:
     - Calculate `S = 1,099,511,627,775 - N`.
     - Encode `S` using Base32 Crockford encoding.
     - Left-pad to 8 characters.
  - **Lexicographical Ordering**: Latest commit has lexicographically smallest name.
- **Branch Reference File Content**:

  ```json
  {
     "manifest_id": "manifest_v{version}"
  }
  ```

**Tag References**:

- **Directory Structure**:

  ```
  refs/
  └── tag.{tag_name}/
        └── ref.json
  ```

  - `{tag_name}`: Name of the tag (no `/` character).

- **Tag Reference File Content**:

  ```json
  {
     "manifest_id": "manifest_v{version}"
  }
  ```

---

### 5. Data Model and Schemas

#### 5.1 Enumerations and Unions

**DeltaType Enumeration**

```flatbuffers
enum DeltaType : byte {
  AddTile = 0,
  ModifyTile = 1,
  DeleteTile = 2,
  UpdateArray = 3,
  UpdateGroup = 4,
  UpdateAttributes = 5
}
```

**CodecParameters Union**

```flatbuffers
union CodecParameters {
  BytesParams
}
```

**DeltaData Union**

```flatbuffers
union DeltaData {
  AddTileData,
  ModifyTileData,
  DeleteTileData,
  UpdateArrayData,
  UpdateGroupData,
  UpdateAttributesData
}
```

#### 5.2 Tables

**KeyValuePair**

```flatbuffers
table KeyValuePair {
  key: string;
  value: string;
}
```

**GroupMetadata**

```flatbuffers
table GroupMetadata {
  group_id: string;
  group_name: string;
  group_path: string;
  attributes: [KeyValuePair];
  created_at: string;
  created_by: string;
}
```

**ArrayMetadata**

```flatbuffers
table ArrayMetadata {
  array_id: string;
  array_name: string;
  group_id: string;
  dtype: string;
  shape: [ulong];
  chunks: [ulong];
  codecs: [Codec];
  attributes: [KeyValuePair];
  created_at: string;
  created_by: string;
}
```

**Codec**

```flatbuffers
table Codec {
  name: string;
  configuration_type: string;
  configuration: CodecParameters;
}
```

**Codec Parameter Tables**

- **BytesParams**

```flatbuffers
table BytesParams {
    endian: string;
}
```

**TileMetadata**

```flatbuffers
table TileMetadata {
  tile_file_path: string;
  array_id: string;
  grid_indices: [ulong];
  shape: [ulong];
  dtype: string;
  codecs: [Codec];
  timestamp: string;
  delta_id: string;
}
```

**Delta**

```flatbuffers
table Delta {
  delta_type: DeltaType;
  delta_data: DeltaData;
}
```

**Transaction**

```flatbuffers
table Transaction {
  transaction_id: string;
  timestamp: string;
  deltas: [Delta];
  branch_name: string;
}
```

**Manifest**

```flatbuffers
table Manifest {
  manifest_id: string;
  version: ulong;
  timestamp: string;
  parent_manifest_ids: [string];
  deltas: [Delta];
  metadata_id: string;
}
```

#### 5.3 FlatBuffers Schemas

All tables and enums defined above are part of the Coral FlatBuffers schema. Implementers should use these definitions to serialize and deserialize data structures used in the Coral format.

### 6. Data Format Logic and Workflow

#### 6.1 Tiling Algorithm

**Objective**: Partition arrays into tiles according to the specified chunk grid.

**Process**:

1. **Define Chunk Sizes**: Use the `chunks` parameter from the array metadata.
2. **Calculate Grid Dimensions**:
    - For each dimension `i`:
      ```
      num_chunks_i = ceil(array_size_i / chunk_size_i)
      ```
3. **Generate Tiles**: Iterate over all combinations of grid indices.
4. **Store Tiles**:
    - Use the naming convention in [Section 4.2](#42-tile-files-and-naming-conventions).
    - Store tiles in the `tiles/` directory.

#### 6.2 Composable Codecs

**Objective**: Use flexible codecs for data compression and transformation.

**Codec Pipeline**:

- **Encoding (Writing)**:
  1. Start with raw tile data.
  2. Apply each codec in the `codecs` list from `ArrayMetadata`.
  3. Store the final output as the tile data.
- **Decoding (Reading)**:
  1. Read the stored tile data.
  2. Apply each codec in reverse order.
  3. Obtain the raw tile data.

#### 6.3 Delta Commit Process

**Steps**:

1. **Identify Changes**: Determine added, modified, or deleted tiles; updates to metadata.
2. **Create Deltas**: For each change, create an appropriate delta.
3. **Store New Content**:
    - **Tiles**: Store in `tiles/` directory.
    - **Metadata**: Include in deltas.
4. **Group Deltas into a Transaction**:
    - Create a `Transaction` object containing all deltas.
5. **Conflict Detection**:
    - Retrieve the latest branch reference.
    - Ensure the branch hasn’t advanced since the transaction was prepared.
6. **Create a New Manifest**:
    - Assign `manifest_id`, increment `version`, set `timestamp`.
    - Reference `parent_manifest_ids`.
    - Include transaction id.
    - Update `recent_metadata_ids`.
7. **Serialize and Store**:
    - Save `Transaction` in `transactions/`.
    - Save `Manifest` in `manifests/`.
8. **Update Branch Reference**:
    - Determine next sequence number `N`.
    - Calculate encoded sequence.
    - Create branch reference file using atomic "create if not exists".

#### 6.4 Dataset Reconstruction

**Objective**: Reconstruct the dataset at a specific version.

**Process**:

1. **Locate Branch Reference**: Find the latest branch reference file.
2. **Retrieve Manifests**: Collect manifests by following `parent_manifest_ids`. If more than one `parent_manifest_id` is present then the manifests are applied in order.
3. **Initialize Dataset State**: Start with an empty state.
4. **Apply Deltas**: Apply deltas in chronological order.
5. **Reconstruct Arrays and Groups**: Assemble dataset using metadata.

### 7. Atomicity and Consistency

#### 7.1 Atomicity

- **Transactions**: Ensure all changes are applied together or not at all.
- **Immutable Files**: Prevent partial updates.
- **Atomic Operations**: Use atomic "create if not exists" for branch references.
- **All-or-Nothing Principle**: If commit fails, the dataset remains unchanged.

#### 7.2 Consistency

- **Version Control**: Each transaction transitions the dataset to a new consistent state.
- **Conflict Detection**: Conflicts are detected via atomic branch updates.
- **Immutable Branch References**: Ensure consistent updates.
- **Data Integrity**: Dataset states are valid and reconstructable.
- **Hierarchy in Deltas**: The dataset’s structure is fully captured.

### 8. Examples

#### 8.1 Array Creation and Reshaping

#### 8.1 Array Creation and Reshaping

**Scenario**: Create an array `temperature_grid` and later reshape it by appending new data.

##### 1. Initial Array Creation

- **Branch**: `main`
- **Sequence Number**: N = 0
- **Encoded Sequence**: `ZZZZZZZZ`

**Process**:

1. **Create Deltas**: As previously described.
2. **Commit Transaction**: Create `Transaction` and `Manifest` objects.
3. **Update Branch Reference**:
    - **File Path**: `refs/branch.main/ZZZZZZZZ.json`
    - **Content**:

      ```json
      {
         "manifest_id": "manifest_v1"
      }
      ```

    - **Atomic Creation**: Use a "create if not exists" operation.

##### 2. Reshaping the Array

- **Branch**: `main`
- **Sequence Number**: N = 1
- **Encoded Sequence**: `ZZZZZZZY`

**Process**:

1. **Create Deltas**: As previously described.
2. **Commit Transaction**: Create `Transaction` and `Manifest` objects.
3. **Update Branch Reference**:
    - **File Path**: `refs/branch.main/ZZZZZZZY.json`
    - **Content**:

      ```json
      {
         "manifest_id": "manifest_v2"
      }
      ```

#### 8.2 Commit and Conflict Resolution

**Commit Process with Immutable Branches**:

1. **Prepare Commit**: User gathers changes and determines current sequence number N.
2. **Create Transaction and Manifest**: As described.
3. **Update Branch Reference**:
    - **Calculate Encoded Sequence**:
      - \( S = 1,099,511,627,775 - (N + 1) \)
      - Encode \( S \) to get `{encoded_sequence}`.
    - **Atomic Creation**:
      - Attempt to create `refs/branch.{branch_name}/{encoded_sequence}.json` using "create if not exists".
      - If the file creation succeeds, the commit is successful.
      - If the file already exists (another commit happened), the commit fails.
4. **Conflict Detection and Resolution**:
    - **Detection**: If the branch reference file for the expected sequence number already exists, a conflict is detected.
    - **Resolution**:
      - The client must retrieve the latest branch reference to get the updated sequence number.
      - Re-prepare the transaction using the new sequence number.
      - Retry the commit process.

### 9. Implementation Considerations

#### 9.1 Storage Backend Compatibility

- **Atomic Operations**:
  - Ensure the storage backend supports atomic "create if not exists" operations.
  - For object stores without native support, implement application-level mechanisms.
- **Consistency Models**:
  - Be aware of eventual consistency and implement retries.
  - Use versioning features if available.
- **File System Semantics**:
  - Understand the storage backend’s handling of file naming, character restrictions, and case sensitivity.

#### 9.2 Data Integrity Verification

- **Verification Mechanisms**:
  - Implement checksums or hashes for data files.
  - Use storage backend features like ETags.
- **Error Handling**:
  - Handle integrity failures gracefully.
  - Implement retry logic for transient errors.

---

### 10. Conclusion

This specification defines the Coral data format with a focus on the required properties of the storage system to ensure atomicity and consistency. By specifying the necessary store capabilities and constraints, implementers can effectively utilize various storage backends while maintaining data integrity.

**Key Points**:

- **Store Requirements**:
  - Atomic file creation and write operations.
  - Consistent namespace and unique file paths.
  - Handling of eventual consistency models.
- **Atomicity and Consistency**:
  - Achieved through immutable files, transactions, and atomic branch updates.
  - Conflict detection via atomic operations ensures consistent dataset states.

**Implementation Guidance**:

- **Adapt to Storage Backend**:
  - Ensure the storage system meets the specified requirements.
  - Implement necessary mechanisms to handle limitations.
- **Robust Error Handling**:
  - Implement data integrity verification.
  - Handle conflicts and retries appropriately.

By adhering to this specification, developers can implement the Coral data format effectively, enabling scalable, reliable, and efficient storage and management of tensor data across various storage backends.

**End of Specification**
