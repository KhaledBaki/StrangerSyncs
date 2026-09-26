from .firebase_database import get_database


def main():
    test_document = None
    created = False

    try:
        db = get_database()

        # Use a separate collection so this test cannot be mistaken
        # for a real person in the future "people" collection.
        test_document = db.collection("connection_tests").document()

        test_document.set({
            "name": "Firebase Test",
            "conversations": 1,
        })
        created = True
        print(f"CREATE worked. Document ID: {test_document.id}")

        saved = test_document.get()
        if not saved.exists:
            raise RuntimeError("The test document could not be read.")

        print(f"READ worked: {saved.to_dict()}")

        test_document.update({"conversations": 2})
        updated = test_document.get().to_dict()

        if updated["conversations"] != 2:
            raise RuntimeError("The update did not save correctly.")

        print(f"UPDATE worked: {updated}")

    except Exception as error:
        print(f"ERROR: Firebase test failed: {error}")

    finally:
        if created:
            try:
                test_document.delete()
                print("CLEANUP worked. Test document deleted.")
            except Exception as error:
                print(f"WARNING: Could not delete test document: {error}")


if __name__ == "__main__":
    main()